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


class TemporalRankingModel(nn.Module):
    def __init__(self, internvl_wrapper, hidden_dim):
        super().__init__()
        self.internvl = internvl_wrapper
        self.ranking_head = RankingHead(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, pixel_values_list, num_patches_lists):
        """
        Extract embeddings and compute ranking scores
        
        Args:
            pixel_values_list: List of [num_patches, 3, H, W] tensors
            num_patches_lists: List of num_patches
        
        Returns:
            scores: [K] ranking scores
        """
        embeddings = []
        
        for pixel_values, num_patches_list in zip(pixel_values_list, num_patches_lists):
            # Process image using InternVL processor
            inputs = self.internvl.process_image_for_forward(
                pixel_values, 
                num_patches_list
            )
            
            # Forward through InternVL
            hidden_states = self.internvl(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Extract last token: [1, seq_len, hidden_dim] -> [hidden_dim]
            last_token = hidden_states[0, -1, :]
            embeddings.append(last_token)
        
        # Stack: [K, hidden_dim]
        embeddings = torch.stack(embeddings, dim=0).unsqueeze(0)  # [1, K, hidden_dim]
        
        # Get ranking scores
        scores = self.ranking_head(embeddings)  # [1, K]
        
        return scores.squeeze(0)  # [K]


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
    
    for batch in pbar:
        pixel_values_list = batch['pixel_values_list']
        num_patches_lists = batch['num_patches_lists']
        gt_order = batch['ground_truth_order'].to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            scores = model(pixel_values_list, num_patches_lists)
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
            print(f"\nError in batch: {e}")
            continue
    
    return total_loss / len(dataloader)


def main():
    args = get_train_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda')
    
    print(f"\n{'='*80}")
    print("InternVL3 Temporal Ranking Training")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Freeze vision: {args.freeze_vision}")
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
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Initialize InternVL wrapper
    internvl_wrapper = InternVL3Wrapper(args.model_path, device=device)
    hidden_dim = internvl_wrapper.hidden_dim
    
    # Setup for training: freeze vision + LoRA
    internvl_wrapper = setup_model_for_training(
        internvl_wrapper,
        freeze_vision=args.freeze_vision,
        use_lora=True
    )
    
    # Create full model
    model = TemporalRankingModel(internvl_wrapper, hidden_dim).to(device)
    
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