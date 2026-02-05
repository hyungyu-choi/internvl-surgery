"""
InternVL3 Model Wrapper using HuggingFace official implementation
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor


class InternVL3Wrapper(nn.Module):
    """
    Wrapper for InternVL3 using HuggingFace official API
    """
    
    def __init__(self, model_path, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        
        print(f"Loading InternVL3 from {model_path}...")
        print(f"Using dtype: {dtype}")
        
        # Load components separately
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        
        self.hidden_dim = self.model.language_model.config.hidden_size
        
        # Get image sequence length from config
        self.image_seq_length = getattr(self.model.config, 'image_seq_length', 256)
        print(f"✓ Model loaded. Hidden dim: {self.hidden_dim}, image_seq_length: {self.image_seq_length}")
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        print("✓ Vision encoder frozen")
    
    def forward(self, **kwargs):
        """
        Forward pass through InternVL model
        
        Args:
            **kwargs: All model inputs
        
        Returns:
            last_hidden_state: [B, seq_len, hidden_dim]
        """
        outputs = self.model(
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # CausalLMOutputWithPast has hidden_states tuple
        # hidden_states[-1] is the last layer
        return outputs.hidden_states[-1]
    
    def process_single_image(self, image):
        """
        Process single PIL image with exact 256 image pad tokens
        
        Args:
            image: PIL Image
        
        Returns:
            Processed inputs dict
        """
        # Process single image
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Use ONLY image_pad tokens (exactly 256)
        image_pad = "<|image_pad|>"
        text = image_pad * self.image_seq_length  # Exactly 256 tokens
        
        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # Create image_flags for single image
        image_flags = torch.tensor([[1]], dtype=torch.long)
        
        # Combine
        inputs = {
            'pixel_values': image_inputs['pixel_values'].to(self.device).to(self.dtype),
            'input_ids': text_inputs['input_ids'].to(self.device),
            'attention_mask': text_inputs['attention_mask'].to(self.device),
            'image_flags': image_flags.to(self.device)
        }
        
        return inputs
    
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