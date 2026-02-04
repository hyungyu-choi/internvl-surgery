# Quick Start Guide

## Installation

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/internvl-surgery.git
cd internvl-surgery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Download Model
The InternVL3 model will be automatically downloaded from HuggingFace on first use.

## Running Inference

### Basic Usage
```bash
# Cholec80 dataset
python inference.py --dataset cholec --gpu 0

# AutoLaparo dataset  
python inference.py --dataset autolaparo --gpu 0
```

### Advanced Options
```bash
python inference.py \
    --dataset cholec \
    --model_path OpenGVLab/InternVL3-8B \
    --gpu 0 \
    --window_size 5 \
    --layer_idx 13 \
    --skip_existing
```

### Custom Paths
```bash
python inference.py \
    --dataset cholec \
    --embeddings_path ./my_embeddings.pkl \
    --frames_dir ./my_frames \
    --output_dir ./my_results
```

## Preparing Your Data

### Directory Structure
```
your_dataset/
├── test_set/
│   ├── 01/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   ├── 02/
│   │   └── ...
│   └── ...
```

### Phase Embeddings Format
```python
{
    'phase_stats': {
        'Phase1': {
            'centroid': np.array([...]),  # [hidden_dim]
            'n_samples': 100
        },
        'Phase2': {
            'centroid': np.array([...]),
            'n_samples': 150
        },
        ...
    }
}
```

## Understanding Arguments

### Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | **required** | Dataset name (cholec/autolaparo) |
| `--model_path` | str | OpenGVLab/InternVL3-8B | Path to InternVL3 model |
| `--layer_idx` | int | 13 | Layer to extract embeddings from |
| `--gpu` | int | 0 | GPU device ID |
| `--window_size` | int | 5 | Temporal smoothing window |
| `--skip_existing` | flag | False | Skip already processed videos |
| `--embeddings_path` | str | None | Override embeddings path |
| `--frames_dir` | str | None | Override frames directory |
| `--output_dir` | str | None | Override output directory |

### Training Arguments (Future)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | **required** | Dataset name |
| `--batch_size` | int | 8 | Batch size per GPU |
| `--epochs` | int | 10 | Number of epochs |
| `--lr` | float | 1e-4 | Learning rate |
| `--freeze_vision` | flag | False | Freeze vision encoder |
| `--freeze_llm` | flag | False | Freeze language model |
| `--wandb` | flag | False | Use W&B logging |

## Output Format

Results are saved as JSON files with the following structure:

```json
[
  {
    "frame": "frame_0001.jpg",
    "index": 1,
    "predicted_phase": "Preparation",
    "confidence": {
      "distance": 2.3456,
      "top_3_candidates": [
        {"phase": "Preparation", "distance": 2.3456},
        {"phase": "ClippingCutting", "distance": 3.1234},
        {"phase": "GallbladderDissection", "distance": 3.8901}
      ]
    },
    "temporal_window": {
      "window_size": 5,
      "window_frames": [1, 2, 3]
    }
  },
  ...
]
```

## Adding a New Dataset

### 1. Create Dataset Class
```python
# data/my_dataset.py
from .base_dataset import SurgicalPhaseDataset
from utils import get_frame_list

class MyDataset(SurgicalPhaseDataset):
    PHASE_NAMES = ['Phase1', 'Phase2', ...]
    
    def _get_frame_list(self):
        return get_frame_list(self.frames_dir)
    
    def get_phase_names(self):
        return self.PHASE_NAMES
```

### 2. Register in Config
```python
# configs/config.py
DATASET_CONFIGS = {
    ...
    'mydataset': {
        'embeddings_path': "path/to/embeddings.pkl",
        'base_frames_dir': "path/to/frames",
        'output_root': "path/to/output",
        'num_phases': 5,
    }
}
```

### 3. Update Factory
```python
# data/__init__.py
from .my_dataset import MyDataset

def get_dataset(dataset_name, **kwargs):
    dataset_map = {
        ...
        'mydataset': MyDataset
    }
    ...
```

## Common Issues

### CUDA Out of Memory
- Reduce `--batch_size`
- Use smaller `--max_patches`
- Process videos one at a time

### Model Download Issues
```bash
# Pre-download the model
from transformers import AutoModel
model = AutoModel.from_pretrained("OpenGVLab/InternVL3-8B")
```

### Custom Model Path
```bash
python inference.py \
    --dataset cholec \
    --model_path /path/to/local/model
```

## Next Steps

1. ✅ Run inference on your dataset
2. ✅ Analyze prediction results
3. 🔄 Implement custom loss function in `train.py`
4. 🔄 Fine-tune the model
5. 🔄 Evaluate on test set

## Getting Help

- Check `STRUCTURE.md` for architecture details
- See `examples.py` for code examples
- Open an issue on GitHub for bugs
