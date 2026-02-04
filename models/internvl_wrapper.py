"""
InternVL3 Model Wrapper for surgical phase recognition
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from configs import VISUAL_TOKEN_START_IDX, N_VISUAL_TOKENS, TARGET_LAYER_IDX


class InternVL3Wrapper:
    """
    Wrapper for InternVL3 model with embedding extraction
    """
    
    def __init__(self, model_path, device='cuda', layer_idx=TARGET_LAYER_IDX):
        """
        Initialize InternVL3 model
        
        Args:
            model_path: Path to pretrained model
            device: Device to load model on
            layer_idx: Layer index to extract embeddings from
        """
        self.model_path = model_path
        self.device = device
        self.layer_idx = layer_idx
        
        print(f"Loading InternVL3 from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval().to(device)
        
        print(f"Model loaded successfully on {device}")
    
    def extract_visual_embeddings(self, pixel_values, num_patches_list):
        """
        Extract visual embeddings from specified LLM layer
        
        Args:
            pixel_values: Preprocessed image tensor
            num_patches_list: List containing number of patches
        
        Returns:
            Frame embedding as numpy array [hidden_dim]
        """
        pixel_values = pixel_values.to(self.device)
        visual_embeddings = []
        
        def hook_fn(module, input, output):
            """Hook to capture visual token embeddings"""
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Extract visual tokens (start_idx to start_idx + n_tokens)
            visual_end_idx = VISUAL_TOKEN_START_IDX + N_VISUAL_TOKENS
            visual_hidden = hidden_states[:, VISUAL_TOKEN_START_IDX:visual_end_idx, :]
            
            visual_embeddings.append(visual_hidden.detach().cpu())
        
        # Register hook on target layer
        hook_handle = self.model.language_model.model.layers[self.layer_idx].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                # Forward pass with minimal text to trigger visual encoding
                dummy_question = "Describe this image."
                _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    dummy_question,
                    generation_config=dict(max_new_tokens=1, do_sample=False),
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )
        finally:
            hook_handle.remove()
        
        if visual_embeddings:
            # Average pool visual tokens to get frame representation
            visual_emb = visual_embeddings[0]  # [batch, n_visual_tokens, hidden_dim]
            frame_embedding = visual_emb.mean(dim=1).squeeze(0)  # [hidden_dim]
            return frame_embedding.float().numpy()
        
        return None
    
    def extract_embeddings_batch(self, pixel_values_list, num_patches_lists):
        """
        Extract embeddings for a batch of images
        
        Args:
            pixel_values_list: List of preprocessed image tensors
            num_patches_lists: List of num_patches_list for each image
        
        Returns:
            List of frame embeddings
        """
        embeddings = []
        for pixel_values, num_patches_list in zip(pixel_values_list, num_patches_lists):
            embedding = self.extract_visual_embeddings(pixel_values, num_patches_list)
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
        return f"InternVL3Wrapper(model_path={self.model_path}, layer_idx={self.layer_idx}, device={self.device})"
