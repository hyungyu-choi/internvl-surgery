"""
AutoLaparo Dataset for surgical phase recognition
"""
import os
from .base_dataset import SurgicalPhaseDataset
from utils import get_frame_list


class AutoLaparoDataset(SurgicalPhaseDataset):
    """
    AutoLaparo surgical phase dataset
    
    Phases:
        0: Preparation
        1: Dividing ligament and peritoneum
        2: Dividing uterine vessels and ligament
        3: Transecting the cervix
        4: Suturing
        5: Washing
        6: Haemostasis and other procedures
        7: Specimen extraction
    """
    
    PHASE_NAMES = [
        'Preparation',
        'Dividing ligament and peritoneum',
        'Dividing uterine vessels and ligament',
        'Transecting the cervix',
        'Suturing',
        'Washing',
        'Haemostasis and other procedures',
        'Specimen extraction'
    ]
    
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
        super().__init__(frames_dir, input_size, max_num_patches,
                        transform, phase_labels)
    
    def _get_frame_list(self):
        """Get sorted list of frame files"""
        return get_frame_list(self.frames_dir, extensions=('.jpg', '.png'))
    
    def get_phase_names(self):
        """Return list of valid phase names"""
        return self.PHASE_NAMES
    
    @staticmethod
    def load_annotations(annotation_file):
        """
        Load phase annotations from file
        
        Args:
            annotation_file: Path to annotation file
        
        Returns:
            Dictionary mapping frame_idx to phase_id
        """
        phase_labels = {}
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found: {annotation_file}")
            return phase_labels
        
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    frame_idx = int(parts[0])
                    phase_id = int(parts[1])
                    phase_labels[frame_idx] = phase_id
        
        return phase_labels
