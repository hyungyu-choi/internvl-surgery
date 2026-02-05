"""
Image preprocessing and transformation utilities for InternVL3
"""
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


def build_transform(input_size=448):
    """
    Build standard InternVL3 image transformation pipeline
    
    Args:
        input_size: Target image size (default: 448)
    
    Returns:
        torchvision.transforms.Compose object
    """
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=1):
    """
    Dynamic preprocessing for InternVL3
    
    Args:
        image: PIL Image
        image_size: Target size
        use_thumbnail: Whether to include thumbnail
        max_num: Maximum number of tiles
    
    Returns:
        List of preprocessed image tiles
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")
    
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    
    tiles = [image]
    if use_thumbnail:
        tiles.append(image.resize((image_size, image_size)))
    
    return tiles[:max_num]


def preprocess_image(image_path, input_size=448, max_num_patches=12):
    """
    Complete preprocessing pipeline for a single image
    
    Args:
        image_path: Path to image file
        input_size: Target image size
        max_num_patches: Maximum number of patches
    
    Returns:
        pixel_values: Preprocessed tensor [1, num_patches, 3, H, W]
        num_patches_list: List containing number of patches
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Dynamic preprocessing
    tiles = dynamic_preprocess(img, image_size=input_size, 
                              use_thumbnail=False, max_num=max_num_patches)
    
    # Apply transforms
    transform = build_transform(input_size)
    pixel_values_list = [transform(tile) for tile in tiles]
    
    # Stack tiles
    pixel_tensor = torch.stack(pixel_values_list)
    num_patches_list = [pixel_tensor.size(0)]
    
    # Add batch dimension if needed
    pixel_values = pixel_tensor.unsqueeze(0) if pixel_tensor.dim() == 3 else pixel_tensor
    
    return pixel_values.to(torch.bfloat16), num_patches_list