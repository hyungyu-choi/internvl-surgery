"""
Training script for finetuning InternVL3 on surgical phase recognition
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_train_args, get_dataset_config
from models import InternVL3Wrapper
from data import get_dataset


class SurgicalPhaseClassifier(nn.Module):
    """
    Surgical phase classifier wrapper around InternVL3
    """
    
    def __init__(self, model_wrapper, num_phases, hidden_dim=4096):
        """
        Args:
            model_wrapper: InternVL3Wrapper instance
            num_phases: Number of surgical phases
            hidden_dim: Hidden dimension of embeddings
        """
        super().__init__()
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.layer_idx = model_wrapper.layer_idx
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_phases)
        
    def forward(self, pixel_values, num_patches_list):
        """
        Forward pass for classification
        
        Args:
            pixel_values: Preprocessed images [B, num_patches, 3, H, W]
            num_patches_list: List of number of patches per image
        
        Returns:
            Logits [B, num_phases]
        """
        # TODO: Implement forward pass
        # Extract visual embeddings from InternVL3
        # Pass through classification head
        raise NotImplementedError("Forward pass to be implemented")
    
    def extract_embeddings(self, pixel_values, num_patches_list):
        """Extract visual embeddings without classification"""
        # TODO: Implement embedding extraction
        raise NotImplementedError("Embedding extraction to be implemented")


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: SurgicalPhaseClassifier
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        num_patches_list = batch['num_patches_list']
        
        # Forward pass
        # TODO: Implement training loop
        # logits = model(pixel_values, num_patches_list)
        # loss = criterion(logits, labels)
        
        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # Update metrics
        # total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': 0.0,  # loss.item()
            'acc': 0.0    # correct / total
        })
    
    # TODO: Return actual metrics
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: SurgicalPhaseClassifier
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Get data
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            num_patches_list = batch['num_patches_list']
            
            # Forward pass
            # TODO: Implement validation
            # logits = model(pixel_values, num_patches_list)
            # loss = criterion(logits, labels)
            
            # Update metrics
            # total_loss += loss.item()
            pass
    
    # TODO: Return actual metrics
    metrics = {
        'val_loss': 0.0,
        'val_acc': 0.0
    }
    
    return metrics


def main():
    """Main training function"""
    args = get_train_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"InternVL3 Surgical Phase Recognition - Training")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Get dataset config
    config = get_dataset_config(args.dataset)
    num_phases = config['num_phases']
    
    # TODO: Load datasets
    print("Loading datasets...")
    # train_dataset = get_dataset(args.dataset, ...)
    # val_dataset = get_dataset(args.dataset, ...)
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers
    # )
    
    # Initialize model
    print("Initializing model...")
    model_wrapper = InternVL3Wrapper(args.model_path, device=device)
    model = SurgicalPhaseClassifier(model_wrapper, num_phases)
    model = model.to(device)
    
    # Freeze components if specified
    if args.freeze_vision:
        print("Freezing vision encoder...")
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
    
    if args.freeze_llm:
        print("Freezing language model...")
        for param in model.model.language_model.parameters():
            param.requires_grad = False
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # TODO: Implement training loop
    print("\nStarting training...")
    print("NOTE: Training loop to be implemented")
    print("This is a skeleton for future finetuning work\n")
    
    # for epoch in range(1, args.epochs + 1):
    #     train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
    #     
    #     if epoch % args.eval_freq == 0:
    #         val_metrics = validate(model, val_loader, criterion, device)
    #         print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
    #     
    #     if epoch % args.save_freq == 0:
    #         # Save checkpoint
    #         pass
    
    print("Training complete!")


if __name__ == "__main__":
    main()
