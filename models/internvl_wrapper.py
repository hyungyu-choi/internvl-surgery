"""
InternVL3 Model Wrapper using HuggingFace official implementation
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class InternVL3Wrapper(nn.Module):
    """
    Wrapper for InternVL3 using HuggingFace official API
    """
    
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.model_path = model_path
        self.device = device
        
        print(f"Loading InternVL3 from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        
        self.hidden_dim = self.model.language_model.config.hidden_size
        print(f"Model loaded. Hidden dim: {self.hidden_dim}")
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        print("✓ Vision encoder frozen")
    
    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Forward pass through InternVL model
        
        Args:
            pixel_values: [B, C, H, W] image tensor
            input_ids: [B, seq_len] token ids
            attention_mask: [B, seq_len] attention mask
        
        Returns:
            last_hidden_state: [B, seq_len, hidden_dim]
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.last_hidden_state
    
    def process_image_for_forward(self, pixel_values, num_patches_list):
        """
        Process image using InternVL processor
        
        Args:
            pixel_values: Preprocessed image tensor
            num_patches_list: List of num patches
        
        Returns:
            Processed inputs dict
        """
        # Convert tensor to PIL Image for processor
        from PIL import Image
        import torchvision.transforms as T
        
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)
        
        # Convert from tensor to PIL
        to_pil = T.ToPILImage()
        images = [to_pil(pixel_values[i]) for i in range(pixel_values.size(0))]
        
        # Use processor
        text = "Image"
        inputs = self.processor(
            images=images[0] if len(images) == 1 else images,
            text=text,
            return_tensors='pt',
            padding=True
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def extract_last_token_embedding(self, pixel_values, num_patches_list):
        """
        Extract last token embedding from LLM
        
        Args:
            pixel_values: Preprocessed image tensor
            num_patches_list: List of num patches
        
        Returns:
            Last token embedding [hidden_dim]
        """
        inputs = self.process_image_for_forward(pixel_values, num_patches_list)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            # Extract last token: [1, seq_len, hidden_dim] -> [hidden_dim]
            last_token = outputs.last_hidden_state[0, -1, :]
        
        return last_token.cpu().numpy()
    
    def get_trainable_parameters(self):
        """Return trainable parameters info"""
        trainable_params = []
        total_params = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()
                trainable_params.append(name)
        
        return {
            'trainable_params': trainable_params,
            'trainable_count': trainable_count,
            'total_params': total_params,
            'trainable_ratio': trainable_count / total_params if total_params > 0 else 0
        }