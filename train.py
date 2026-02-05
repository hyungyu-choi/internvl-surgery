import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_train_args, get_dataset_config
from models.internvl_wrapper import InternVL3Wrapper
from models.ranking_head import RankingHead
from losses.plackett_luce import PlackettLuceLoss
from data.cholec_ranking_dataset import CholecRankingDataset


class TemporalRankingModel(nn.Module):
    def __init__(self, internvl_wrapper, hidden_dim=3584):
        super().__init__()
        self.internvl = internvl_wrapper
        self.ranking_head = RankingHead(hidden_dim=hidden_dim)
    
    def forward(self, pixel_values_list, num_patches_lists):
        """
        Extract last tokens and compute ranking scores
        
        Args:
            pixel_values_list: List of [1, num_patches, 3, H, W] tensors
            num_patches_lists: List of num_patches for each image
        
        Returns:
            scores: [K] ranking scores for K frames
        """
        embeddings = []
        
        for pixel_values, num_patches_list in zip(pixel_values_list, num_patches_lists):
            # Use wrapper's extract_last_token method with return_tensor=True
            last_token_tensor = self.internvl.extract_last_token(
                pixel_values, 
                num_patches_list, 
                return_tensor=True
            )
            embeddings.append(last_token_tensor)
        
        # Stack: [K, hidden_dim]
        embeddings = torch.stack(embeddings, dim=0).unsqueeze(0)  # [1, K, hidden_dim]
        
        # Get ranking scores
        scores = self.ranking_head(embeddings)  # [1, K]
        
        return scores.squeeze(0)  # [K]


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.ranking_head.train()
    
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for i, batch in enumerate(pbar):
        pixel_values_list = batch['pixel_values_list']
        num_patches_lists = batch['num_patches_lists']
        gt_order = batch['ground_truth_order'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        scores = model(pixel_values_list, num_patches_lists)
        scores = scores.unsqueeze(0)  # [1, K]
        
        # Compute loss
        loss = criterion(scores, gt_order)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def main():
    args = get_train_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda')
    
    print(f"\n{'='*60}")
    print(f"InternVL3 Temporal Ranking Training")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset)
    
    # Create dataset
    train_dataset = CholecRankingDataset(
        base_frames_dir=config['base_frames_dir'].replace('test_set', 'training_set'),
        num_frames=3,
        samples_per_video=10
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize model
    internvl_wrapper = InternVL3Wrapper(args.model_path, device=device)
    model = TemporalRankingModel(internvl_wrapper).to(device)
    
    # Freeze InternVL3
    for param in model.internvl.model.parameters():
        param.requires_grad = False
    
    # Optimizer for ranking head only
    optimizer = torch.optim.AdamW(
        model.ranking_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = PlackettLuceLoss()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'ranking_head_state_dict': model.ranking_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f"{args.output_dir}/checkpoint_epoch_{epoch}.pth")
            print(f"Saved checkpoint")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()