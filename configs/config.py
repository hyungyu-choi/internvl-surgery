"""
Configuration management for InternVL3 surgical phase recognition
"""
import argparse
import os


# InternVL3 Model Constants
INPUT_SIZE = 448
MAX_NUM_PATCHES = 12

# Dataset Configurations
DATASET_CONFIGS = {
    'cholec': {
        'embeddings_path': "./experiments/embeddings/cholec_internvl3_phase_embeddings_13_layer_5_shot/phase_embeddings_stats.pkl",
        'base_frames_dir': "../../code/Dataset/cholec80/frames/extract_1fps/test_set",
        'output_root': "./experiments/predictions/cholec_internvl3_embedding_13_layer_5_shot_pred_window5_avg_all_testset",
        'num_phases': 7,
    },
    'autolaparo': {
        'embeddings_path': "./experiments/embeddings/autolaparo_internvl3_phase_embeddings_13_layer_5_shot/phase_embeddings_stats.pkl",
        'base_frames_dir': "../../code/Dataset/AutoLaparo_Task1/frames/test_set",
        'output_root': "./experiments/predictions/autolaparo_internvl3_embedding_13_layer_5_shot_pred_window5_avg_all_testset",
        'num_phases': 8,
    }
}


def get_inference_args():
    """Parse arguments for inference"""
    parser = argparse.ArgumentParser('InternVL3 Surgical Phase Inference', add_help=True)
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cholec', 'autolaparo'],
                        help='Dataset to use (cholec or autolaparo)')
    
    # Model
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL3-8B',
                        help='Path to InternVL3 model')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    # Inference
    parser.add_argument('--window_size', type=int, default=5,
                        help='Temporal window size for smoothing')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have predictions')
    
    # Paths (optional overrides)
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Override embeddings path')
    parser.add_argument('--frames_dir', type=str, default=None,
                        help='Override frames directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    
    return parser.parse_args()


def get_train_args():
    """Parse arguments for training"""
    parser = argparse.ArgumentParser('InternVL3 Surgical Phase Training', add_help=True)
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cholec', 'autolaparo'],
                        help='Dataset to use')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Training split name')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Validation split name')
    
    # Model
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL3-8B',
                        help='Path to pretrained InternVL3')
    parser.add_argument('--freeze_vision', action='store_true',
                        help='Freeze vision encoder')
    parser.add_argument('--freeze_llm', action='store_true',
                        help='Freeze language model')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    
    # Image
    parser.add_argument('--image_size', type=int, default=INPUT_SIZE,
                        help='Input image size')
    parser.add_argument('--max_patches', type=int, default=MAX_NUM_PATCHES,
                        help='Maximum number of patches')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluate every N epochs')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='internvl-surgery',
                        help='W&B project name')
    
    return parser.parse_args()


def get_dataset_config(dataset_name, args=None):
    """Get dataset configuration with optional overrides"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name].copy()
    
    # Apply overrides if provided
    if args:
        if args.embeddings_path:
            config['embeddings_path'] = args.embeddings_path
        if args.frames_dir:
            config['base_frames_dir'] = args.frames_dir
        if args.output_dir:
            config['output_root'] = args.output_dir
    
    return config