import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from configs import get_train_args, get_dataset_config
from models.internvl_wrapper import InternVL3Wrapper
from models.ranking_head import RankingHead
from losses.plackett_luce import PlackettLuceLoss
from data.cholec_ranking_dataset import CholecRankingDataset

def collate_fn(batch):
    """Custom collate function for PIL images"""
    images = [item['images'] for item in batch]
    gt_orders = torch.stack([item['ground_truth_order'] for item in batch])
    video_ids = [item['video_id'] for item in batch]
    
    return {
        'images': images,  # List of lists of PIL images
        'ground_truth_order': gt_orders,
        'video_ids': video_ids
    }

class TemporalRankingModel(nn.Module):
    def __init__(self, internvl_wrapper, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        self.internvl = internvl_wrapper
        self.ranking_head = RankingHead(hidden_dim=hidden_dim, dtype=dtype)
        self.hidden_dim = hidden_dim
        self.dtype = dtype
    
    def forward(self, images_list):
        """
        Extract embeddings and compute ranking scores
        
        Args:
            images_list: List of PIL Images [K images]
        
        Returns:
            scores: [K] ranking scores
        """
        embeddings = []
        
        for idx, img in enumerate(images_list):
            # Process SINGLE image
            inputs = self.internvl.process_single_image(img)
            
            # Forward through InternVL
            with torch.no_grad():  # Save memory
                hidden_states = self.internvl(**inputs)
            
            # Extract last token: [1, seq_len, hidden_dim] -> [hidden_dim]
            last_token = hidden_states[0, -1, :].clone()
            embeddings.append(last_token)
        
        # Stack and get scores
        embeddings = torch.stack(embeddings, dim=0).unsqueeze(0)  # [1, K, hidden_dim]
        embeddings = embeddings.to(self.dtype)
        scores = self.ranking_head(embeddings)  # [1, K]
        
        return scores.squeeze(0)  # [K]

def check_model_dtypes(model, verbose=True):
    """Check and report dtypes across the model"""
    if verbose:
        print(f"\n{'='*80}")
        print("DTYPE CHECK")
        print(f"{'='*80}")
    
    # Check InternVL
    internvl_dtypes = {}
    for name, param in model.internvl.model.named_parameters():
        dtype = param.dtype
        if dtype not in internvl_dtypes:
            internvl_dtypes[dtype] = 0
        internvl_dtypes[dtype] += 1
    
    if verbose:
        print(f"\nInternVL Model:")
        for dtype, count in internvl_dtypes.items():
            print(f"  {dtype}: {count} parameters")
    
    # Check RankingHead
    ranking_dtypes = {}
    for name, param in model.ranking_head.named_parameters():
        dtype = param.dtype
        if dtype not in ranking_dtypes:
            ranking_dtypes[dtype] = 0
        ranking_dtypes[dtype] += 1
    
    if verbose:
        print(f"\nRanking Head:")
        for dtype, count in ranking_dtypes.items():
            print(f"  {dtype}: {count} parameters")
        print(f"{'='*80}\n")
    
    return internvl_dtypes, ranking_dtypes


def print_trainable_parameters(model):
    """Print detailed trainable parameters info"""
    print(f"\n{'='*80}")
    print("TRAINABLE PARAMETERS")
    print(f"{'='*80}")
    
    # InternVL parameters
    internvl_info = model.internvl.get_trainable_parameters()
    print(f"\nInternVL Model:")
    print(f"  Total params: {internvl_info['total_params']:,}")
    print(f"  Trainable params: {internvl_info['trainable_count']:,}")
    print(f"  Trainable ratio: {internvl_info['trainable_ratio']:.2%}")
    
    if internvl_info['trainable_params']:
        print(f"\n  Trainable layers (first 20):")
        for name in internvl_info['trainable_params'][:20]:
            print(f"    ✓ {name}")
        if len(internvl_info['trainable_params']) > 20:
            print(f"    ... and {len(internvl_info['trainable_params']) - 20} more")
    else:
        print(f"\n  No trainable layers in InternVL")
    
    # Ranking head parameters
    ranking_params = sum(p.numel() for p in model.ranking_head.parameters() if p.requires_grad)
    print(f"\nRanking Head:")
    print(f"  Trainable params: {ranking_params:,}")
    
    # Total
    total_trainable = internvl_info['trainable_count'] + ranking_params
    total_params = internvl_info['total_params'] + sum(p.numel() for p in model.ranking_head.parameters())
    print(f"\nTotal Model:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"  Trainable ratio: {total_trainable/total_params:.2%}")
    print(f"{'='*80}\n")


def apply_lora_to_llm(model, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """Apply LoRA to language model"""
    print("\nApplying LoRA to language model...")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.language_model = get_peft_model(model.language_model, lora_config)
    
    # Convert LoRA parameters to bfloat16
    for name, param in model.language_model.named_parameters():
        if param.requires_grad and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    
    print("✓ LoRA applied to language model")
    
    return model


def setup_model_for_training(wrapper, freeze_vision=True, use_lora=True):
    """Setup model for training: freeze vision encoder, apply LoRA"""
    
    # 1. Freeze vision encoder
    if freeze_vision:
        wrapper.freeze_vision_encoder()
    
    # 2. Apply LoRA to language model
    if use_lora:
        wrapper.model = apply_lora_to_llm(wrapper.model)
    
    # 3. Ensure projection layer (mlp1) is trainable
    for param in wrapper.model.mlp1.parameters():
        param.requires_grad = True
    print("✓ Projection layer (mlp1) set to trainable")
    
    return wrapper


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    model.ranking_head.train()
    
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # batch['images'] is list of list: [[img1, img2, img3], ...]
        # Since batch_size=1, take first element
        images_list = batch['images'][0]  # This is [img1, img2, img3]
        gt_order = batch['ground_truth_order'][0:1]  # Keep batch dimension: [1, K]
        gt_order = gt_order.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass - images_list is a list of PIL images
            scores = model(images_list)  # Returns [K]
            scores = scores.unsqueeze(0)  # [1, K]
            
            # Compute loss
            loss = criterion(scores, gt_order)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        except Exception as e:
            print(f"\n[ERROR] Batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            if batch_idx == 0:
                print(f"\nDebug info:")
                print(f"  type(images_list): {type(images_list)}")
                print(f"  len(images_list): {len(images_list)}")
                print(f"  type(images_list[0]): {type(images_list[0])}")
            raise e
    
    return total_loss / len(dataloader)

def main():
    args = get_train_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda')
    dtype = torch.bfloat16
    
    print(f"\n{'='*80}")
    print("InternVL3 Temporal Ranking Training")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Freeze vision: {args.freeze_vision}")
    print(f"Dtype: {dtype}")
    print(f"{'='*80}\n")
    
    config = get_dataset_config(args.dataset)
    
    # Create dataset
    train_dataset = CholecRankingDataset(
        base_frames_dir=config['base_frames_dir'].replace('test_set', 'training_set'),
        num_frames=3,
        samples_per_video=10
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn  # ✓ 추가
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Initialize InternVL wrapper with explicit dtype
    internvl_wrapper = InternVL3Wrapper(
        args.model_path, 
        device=device,
        dtype=dtype
    )
    hidden_dim = internvl_wrapper.hidden_dim
    
    # Setup for training: freeze vision + LoRA
    internvl_wrapper = setup_model_for_training(
        internvl_wrapper,
        freeze_vision=args.freeze_vision,
        use_lora=True
    )
    
    # Create full model with matching dtype
    model = TemporalRankingModel(
        internvl_wrapper, 
        hidden_dim,
        dtype=dtype
    ).to(device)
    
    # Check dtype consistency
    print("\n" + "="*80)
    print("Initial dtype check:")
    check_model_dtypes(model, verbose=True)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = PlackettLuceLoss()
    
    # Training loop
    print(f"{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            
            save_path = f"{args.output_dir}/checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, save_path)
            print(f"  ✓ Checkpoint saved: {save_path}")
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()