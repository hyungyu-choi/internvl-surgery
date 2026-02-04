"""
Helper utility functions
"""
import os
import re
import json
import numpy as np
from pathlib import Path


def extract_frame_index(filename):
    """
    Extract frame index from filename
    
    Args:
        filename: Frame filename (e.g., 'frame_0001.jpg')
    
    Returns:
        Frame index as integer, -1 if not found
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


def get_video_list(base_path):
    """
    Get list of video IDs from directory
    
    Args:
        base_path: Base directory containing video folders
    
    Returns:
        Sorted list of video IDs
    """
    base_dir = Path(base_path)
    video_list = []
    
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            video_list.append(int(folder.name))
    
    video_list.sort()
    return video_list


def get_frame_list(frames_dir, extensions=('.jpg', '.png')):
    """
    Get sorted list of frame files in directory
    
    Args:
        frames_dir: Directory containing frames
        extensions: Tuple of valid image extensions
    
    Returns:
        Sorted list of frame filenames
    """
    frame_files = [
        f for f in os.listdir(frames_dir) 
        if f.lower().endswith(extensions)
    ]
    return sorted(frame_files, key=extract_frame_index)


def compute_centroid_distance(embedding, centroid):
    """
    Compute L2 distance between embedding and centroid
    
    Args:
        embedding: Feature embedding (numpy array)
        centroid: Phase centroid (numpy array)
    
    Returns:
        L2 distance as float
    """
    return float(np.linalg.norm(embedding - centroid))


def predict_phase_from_embedding(frame_embedding, phase_stats):
    """
    Predict surgical phase using nearest centroid
    
    Args:
        frame_embedding: Frame feature embedding
        phase_stats: Dictionary of phase statistics with centroids
    
    Returns:
        predicted_phase: Predicted phase name
        distances: Dictionary of distances to all phases
        sorted_distances: List of (phase, distance) tuples sorted by distance
    """
    distances = {}
    
    for phase, stats in phase_stats.items():
        centroid = stats['centroid']
        distance = compute_centroid_distance(frame_embedding, centroid)
        distances[phase] = distance
    
    # Find nearest phase
    predicted_phase = min(distances, key=distances.get)
    
    # Sort by distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    
    return predicted_phase, distances, sorted_distances


def apply_temporal_smoothing(embeddings, window_size=5):
    """
    Apply temporal window averaging to embeddings
    
    Args:
        embeddings: List of frame embeddings
        window_size: Size of temporal window
    
    Returns:
        List of smoothed embeddings
    """
    half_window = window_size // 2
    smoothed = []
    
    for i in range(len(embeddings)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(embeddings), i + half_window + 1)
        
        window_embeddings = embeddings[start_idx:end_idx]
        averaged = np.mean(window_embeddings, axis=0)
        smoothed.append(averaged)
    
    return smoothed


def save_predictions(results, output_path):
    """
    Save prediction results to JSON file
    
    Args:
        results: List of prediction dictionaries
        output_path: Output JSON file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_predictions(json_path):
    """
    Load predictions from JSON file
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        List of prediction dictionaries
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def check_existing_predictions(output_dir):
    """
    Find existing prediction files
    
    Args:
        output_dir: Directory to check
    
    Returns:
        Set of video IDs that have predictions
    """
    if not os.path.exists(output_dir):
        return set()
    
    existing = set()
    for fname in os.listdir(output_dir):
        match = re.match(r"(\d+)\.json$", fname)
        if match:
            existing.add(int(match.group(1)))
    
    return existing
