import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils import get_frame_list, preprocess_image


class CholecRankingDataset(Dataset):
    def __init__(self, base_frames_dir, num_frames=5, samples_per_video=10):
        self.base_frames_dir = Path(base_frames_dir)
        self.num_frames = num_frames
        self.samples_per_video = samples_per_video
        
        # Get video list (1-40 for train)
        self.video_ids = sorted([
            int(d.name) for d in self.base_frames_dir.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
        
        self.total_samples = len(self.video_ids) * samples_per_video
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        video_idx = idx // self.samples_per_video
        video_id = self.video_ids[video_idx]
        frames_dir = self.base_frames_dir / f"{video_id:02d}"
        
        # Get all frames
        frame_files = get_frame_list(str(frames_dir))
        
        # Random sample K frames
        sampled_indices = sorted(random.sample(range(len(frame_files)), self.num_frames))
        sampled_frames = [frame_files[i] for i in sampled_indices]
        
        # Preprocess frames
        pixel_values_list = []
        num_patches_lists = []
        
        for frame_file in sampled_frames:
            frame_path = frames_dir / frame_file
            pixel_values, num_patches_list = preprocess_image(str(frame_path))
            pixel_values_list.append(pixel_values)
            num_patches_lists.append(num_patches_list)
        
        # Ground truth order is [0, 1, 2, 3, 4] (already sorted)
        gt_order = torch.arange(self.num_frames, dtype=torch.long)
        
        return {
            'pixel_values_list': pixel_values_list,
            'num_patches_lists': num_patches_lists,
            'ground_truth_order': gt_order,
            'video_id': video_id
        }