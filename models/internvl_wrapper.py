"""
InternVL3 Model Wrapper for surgical phase recognition
Extracts last token from final LLM layer
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class InternVL3Wrapper:
    """
    Wrapper for InternVL3 model with last token extraction
    """
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize InternVL3 model
        
        Args:
            model_path: Path to pretrained model
            device: Device to load model on
        """
        self.model_path = model_path
        self.device = device
        
        print(f"Loading InternVL3 from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval().to(device)
        
        # Get last layer index
        self.last_layer_idx = len(self.model.language_model.model.layers) - 1
        
        print(f"Model loaded successfully on {device}")
        print(f"Using last layer (index {self.last_layer_idx}) for token extraction")
    
    def extract_last_token(self, pixel_values, num_patches_list, return_tensor=False):
        """
        Extract last token from the final LLM layer
        
        Args:
            pixel_values: Preprocessed image tensor
            num_patches_list: List containing number of patches
            return_tensor: If True, return torch tensor on device; if False, return numpy array
        
        Returns:
            Last token embedding as numpy array [hidden_dim] or torch tensor [hidden_dim]
        """
        # Remove batch dimension if present: [1, num_patches, 3, H, W] -> [num_patches, 3, H, W]
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)
        
        pixel_values = pixel_values.to(self.device)
        last_tokens = []
        
        def hook_fn(module, input, output):
            """Hook to capture last token from final layer"""
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Extract last token: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
            last_token = hidden_states[:, -1, :]
            last_tokens.append(last_token.detach().cpu())
        
        # Register hook on last layer
        hook_handle = self.model.language_model.model.layers[self.last_layer_idx].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                # Forward pass with minimal text to trigger encoding
                _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    "Describe this image.",
                    generation_config=dict(max_new_tokens=1, do_sample=False),
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )
        finally:
            hook_handle.remove()
        
        if last_tokens:
            # Get last token
            last_token = last_tokens[0].squeeze(0)  # [hidden_dim]
            
            if return_tensor:
                # Return as float tensor on device
                return last_token.float().to(self.device)
            else:
                # Return as numpy array
                return last_token.float().numpy()
        
        return None
    
    def extract_last_tokens_batch(self, pixel_values_list, num_patches_lists, return_tensor=False):
        """
        Extract last tokens for a batch of images
        
        Args:
            pixel_values_list: List of preprocessed image tensors
            num_patches_lists: List of num_patches_list for each image
            return_tensor: If True, return torch tensors; if False, return numpy arrays
        
        Returns:
            List of last token embeddings (tensors or numpy arrays)
        """
        embeddings = []
        for pixel_values, num_patches_list in zip(pixel_values_list, num_patches_lists):
            embedding = self.extract_last_token(pixel_values, num_patches_list, return_tensor=return_tensor)
            embeddings.append(embedding)
        return embeddings
    
    def chat(self, pixel_values, question, num_patches_list, **kwargs):
        """
        Direct chat interface (for custom prompting)
        
        Args:
            pixel_values: Preprocessed image tensor
            question: Text prompt
            num_patches_list: List containing number of patches
            **kwargs: Additional arguments for generation
        
        Returns:
            Model response
        """
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                num_patches_list=num_patches_list,
                **kwargs
            )
        
        return response
    
    def __repr__(self):
        return f"InternVL3Wrapper(model_path={self.model_path}, device={self.device})"