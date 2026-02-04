"""
Inference script for surgical phase recognition using InternVL3
"""
import os
import json
import pickle
from tqdm import tqdm

from configs import get_inference_args, get_dataset_config
from models import InternVL3Wrapper
from utils import (
    get_video_list,
    get_frame_list,
    predict_phase_from_embedding,
    apply_temporal_smoothing,
    save_predictions,
    check_existing_predictions,
    preprocess_image
)


def infer_video(model, frames_dir, phase_stats, window_size=5):
    """
    Infer surgical phases for a single video
    
    Args:
        model: InternVL3Wrapper instance
        frames_dir: Directory containing video frames
        phase_stats: Phase embedding statistics
        window_size: Temporal smoothing window size
    
    Returns:
        List of prediction results
    """
    # Get frame list
    frame_files = get_frame_list(frames_dir)
    
    # Step 1: Extract embeddings for all frames
    print(f"Extracting embeddings for {len(frame_files)} frames...")
    frame_embeddings = []
    
    for frame_file in tqdm(frame_files, desc="Extracting embeddings"):
        frame_path = os.path.join(frames_dir, frame_file)
        
        # Preprocess and extract embeddings
        pixel_values, num_patches_list = preprocess_image(frame_path)
        frame_embedding = model.extract_visual_embeddings(pixel_values, num_patches_list)
        
        if frame_embedding is None:
            print(f"Error: Failed to extract embedding for {frame_file}")
            return [{
                "frame": frame_file,
                "index": len(frame_embeddings) + 1,
                "predicted_phase": "Unknown",
                "reason": "Failed to extract visual embeddings",
                "error": True
            }]
        
        frame_embeddings.append(frame_embedding)
    
    # Step 2: Apply temporal smoothing
    print(f"Applying temporal smoothing (window_size={window_size})...")
    smoothed_embeddings = apply_temporal_smoothing(frame_embeddings, window_size)
    
    # Step 3: Predict phases
    print("Predicting phases...")
    results = []
    
    for i, (frame_file, smoothed_emb) in enumerate(zip(frame_files, smoothed_embeddings)):
        # Predict using smoothed embedding
        predicted_phase, distances, sorted_distances = predict_phase_from_embedding(
            smoothed_emb, phase_stats
        )
        
        # Get window range
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(len(frame_embeddings), i + half_window + 1)
        
        # Create result
        top_3_phases = sorted_distances[:3]
        
        result = {
            "frame": frame_file,
            "index": i + 1,
            "predicted_phase": predicted_phase,
            "confidence": {
                "distance": float(distances[predicted_phase]),
                "top_3_candidates": [
                    {"phase": phase, "distance": float(dist)}
                    for phase, dist in top_3_phases
                ]
            },
            "temporal_window": {
                "window_size": window_size,
                "window_frames": list(range(start_idx + 1, end_idx + 1))
            }
        }
        
        results.append(result)
    
    return results


def main():
    """Main inference function"""
    args = get_inference_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Get dataset configuration
    config = get_dataset_config(args.dataset, args)
    
    print(f"\n{'='*60}")
    print(f"InternVL3 Surgical Phase Recognition - Inference")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"Layer: {args.layer_idx}")
    print(f"Window size: {args.window_size}")
    print(f"{'='*60}\n")
    
    # Load phase embeddings
    embeddings_path = config['embeddings_path']
    print(f"Loading phase embeddings from: {embeddings_path}")
    
    if not os.path.exists(embeddings_path):
        print(f"Error: Phase embeddings not found at {embeddings_path}")
        print("Please run the embedding extraction script first.")
        return
    
    with open(embeddings_path, 'rb') as f:
        embedding_data = pickle.load(f)
    
    phase_stats = embedding_data['phase_stats']
    
    print(f"\nLoaded embeddings for {len(phase_stats)} phases:")
    for phase, stats in phase_stats.items():
        print(f"  - {phase}: {stats['n_samples']} samples")
    
    # Initialize model
    model = InternVL3Wrapper(
        model_path=args.model_path,
        device='cuda',
        layer_idx=args.layer_idx
    )
    
    # Get test videos
    base_frames_dir = config['base_frames_dir']
    output_root = config['output_root']
    os.makedirs(output_root, exist_ok=True)
    
    # Check existing predictions
    existing_predictions = check_existing_predictions(output_root)
    
    # Get video list
    video_list = get_video_list(base_frames_dir)
    
    print(f"\nFound {len(video_list)} test videos")
    if args.skip_existing and existing_predictions:
        print(f"Skipping {len(existing_predictions)} videos with existing predictions")
    
    # Process each video
    for video_id in video_list:
        if args.skip_existing and video_id in existing_predictions:
            print(f"\nSkip video {video_id:02d}: already processed")
            continue
        
        frames_dir = os.path.join(base_frames_dir, f"{video_id:02d}")
        output_path = os.path.join(output_root, f"{video_id}.json")
        
        if not os.path.exists(frames_dir):
            print(f"\nSkip video {video_id:02d}: directory not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing video {video_id:02d}")
        print(f"{'='*60}")
        
        # Run inference
        results = infer_video(
            model=model,
            frames_dir=frames_dir,
            phase_stats=phase_stats,
            window_size=args.window_size
        )
        
        # Save results
        save_predictions(results, output_path)
        print(f"Saved predictions to: {output_path}")
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
