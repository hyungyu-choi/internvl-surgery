# configs/config.py
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
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cholec', 'autolaparo'],
                        help='Dataset to use')
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL3-8B',
                        help='Path to InternVL3 model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Temporal window size')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip existing predictions')
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
    
    # Model
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL3-8B',
                        help='Path to pretrained InternVL3')
    parser.add_argument('--freeze_vision', action='store_true', default=True,
                        help='Freeze vision encoder')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()

def get_dataset_config(dataset_name, args=None):
    """Get dataset configuration with optional overrides"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name].copy()
    
    if args:
        if hasattr(args, 'embeddings_path') and args.embeddings_path:
            config['embeddings_path'] = args.embeddings_path
        if hasattr(args, 'frames_dir') and args.frames_dir:
            config['base_frames_dir'] = args.frames_dir
        if hasattr(args, 'output_dir') and args.output_dir:
            config['output_root'] = args.output_dir
    
    return config