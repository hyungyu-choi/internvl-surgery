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
    
    def extract_last_token(self, pixel_values, num_patches_list):
        last_tokens = []
        
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            last_token = hidden_states[:, -1, :]
            last_tokens.append(last_token)
        
        last_layer_idx = len(self.internvl.model.language_model.model.layers) - 1
        hook = self.internvl.model.language_model.model.layers[last_layer_idx].register_forward_hook(hook_fn)
        
        try:
                # Remove batch dimension: [1, num_patches, 3, H, W] -> [num_patches, 3, H, W]
                pixel_values = pixel_values.squeeze(0).to(self.internvl.device)
                
                _ = self.internvl.model.chat(
                    self.internvl.tokenizer,
                    pixel_values,
                    "Describe this image.",
                    generation_config=dict(max_new_tokens=1, do_sample=False),
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )
        finally:
            hook.remove()
        
        return last_tokens[0] if last_tokens else None
    
    def forward(self, pixel_values_list, num_patches_lists):
        embeddings = []
        for pixel_values, num_patches_list in zip(pixel_values_list, num_patches_lists):
            last_token = self.extract_last_token(pixel_values, num_patches_list)
            embeddings.append(last_token.squeeze(0).float())  # bfloat16 -> float32
        
        embeddings = torch.stack(embeddings, dim=0).unsqueeze(0)
        scores = self.ranking_head(embeddings)
        return scores.squeeze(0)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.ranking_head.train()
    
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    # 첫 번째 파라미터 추적
    first_param = next(model.ranking_head.parameters())
    param_before = first_param.clone().detach()
    
    for i, batch in enumerate(pbar):
        pixel_values_list = batch['pixel_values_list']
        num_patches_lists = batch['num_patches_lists']
        gt_order = batch['ground_truth_order'].to(device)
        
        optimizer.zero_grad()
        
        scores = model(pixel_values_list, num_patches_lists)
        scores = scores.unsqueeze(0)
        
        loss = criterion(scores, gt_order)
        
        loss.backward()
        
        # # Gradient 확인
        # if i < 2000:
        #     print(f"\nScores: {scores}")
        #     print(f"GT Order: {gt_order}")
        #     print(f"Loss: {loss.item()}")
        #     has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
        #                   for p in model.ranking_head.parameters())
        #     print(f"Has gradient: {has_grad}")
        #     if has_grad:
        #         grad_norm = sum(p.grad.norm().item() for p in model.ranking_head.parameters() if p.grad is not None)
        #         print(f"Gradient norm: {grad_norm}")
        
        # optimizer.step()
        
        # # 파라미터 변화 확인
        # if i < 2000:
        #     param_after = first_param.clone().detach()
        #     param_diff = (param_after - param_before).abs().sum()
        #     print(f"Parameter change: {param_diff.item()}\n")
        
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
    
    internvl_wrapper = InternVL3Wrapper(args.model_path, device=device)
    model = TemporalRankingModel(internvl_wrapper).to(device)
    
    for param in model.internvl.model.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        model.ranking_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = PlackettLuceLoss()
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
        
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