"""
Base dataset class for surgical phase recognition
"""
import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from PIL import Image

from utils import preprocess_image


class SurgicalPhaseDataset(Dataset, ABC):
    """
    Abstract base class for surgical phase datasets
    """
    
    def __init__(self, frames_dir, input_size=448, max_num_patches=12, 
                 transform=None, phase_labels=None):
        """
        Args:
            frames_dir: Directory containing video frames
            input_size: Input image size
            max_num_patches: Maximum number of patches
            transform: Optional custom transform
            phase_labels: Dictionary mapping frame_idx to phase label
        """
        self.frames_dir = frames_dir
        self.input_size = input_size
        self.max_num_patches = max_num_patches
        self.transform = transform
        self.phase_labels = phase_labels or {}
        
        # Get frame list
        self.frame_files = self._get_frame_list()
        
        print(f"Loaded {len(self.frame_files)} frames from {frames_dir}")
    
    @abstractmethod
    def _get_frame_list(self):
        """Get sorted list of frame files - must be implemented by subclass"""
        pass
    
    @abstractmethod
    def get_phase_names(self):
        """Return list of valid phase names - must be implemented by subclass"""
        pass
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - pixel_values: Preprocessed image tensor
                - num_patches_list: Number of patches
                - frame_path: Path to frame
                - frame_idx: Frame index
                - label: Phase label (if available)
        """
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frames_dir, frame_file)
        
        # Preprocess image
        if self.transform:
            # Custom transform
            pixel_values, num_patches_list = self.transform(frame_path)
        else:
            # Default InternVL3 preprocessing
            pixel_values, num_patches_list = preprocess_image(
                frame_path, self.input_size, self.max_num_patches
            )
        
        # Get label if available
        label = self.phase_labels.get(idx, -1)  # -1 for unlabeled
        
        return {
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,
            'frame_path': frame_path,
            'frame_idx': idx,
            'label': label
        }
    
    def get_frame_path(self, idx):
        """Get path to frame at index"""
        return os.path.join(self.frames_dir, self.frame_files[idx])
