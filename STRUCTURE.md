# Project Structure Documentation

## Overview
This project provides a clean, modular implementation of surgical phase recognition using InternVL3, organized following best practices for deep learning research projects.

## Directory Structure

```
internvl_surgery/
│
├── configs/                    # Configuration management
│   ├── __init__.py
│   └── config.py              # Argument parsing, dataset configs
│
├── models/                     # Model definitions
│   ├── __init__.py
│   └── internvl_wrapper.py    # InternVL3 wrapper with embedding extraction
│
├── data/                       # Dataset implementations
│   ├── __init__.py
│   ├── base_dataset.py        # Abstract base dataset class
│   ├── cholec_dataset.py      # Cholec80 dataset
│   └── autolaparo_dataset.py  # AutoLaparo dataset
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── transforms.py          # Image preprocessing
│   └── helpers.py             # Helper functions
│
├── inference.py               # Main inference script
├── train.py                   # Training script (skeleton for future work)
├── examples.py                # Usage examples
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── .gitignore                # Git ignore rules
└── README.md                  # Project documentation
```

## Module Descriptions

### configs/
- **config.py**: Centralized configuration management
  - `get_inference_args()`: Parse inference arguments
  - `get_train_args()`: Parse training arguments
  - `get_dataset_config()`: Get dataset-specific configuration
  - Constants: Model parameters, dataset paths

### models/
- **internvl_wrapper.py**: Clean wrapper around InternVL3
  - `InternVL3Wrapper`: Main model class
    - `extract_visual_embeddings()`: Extract embeddings from specified layer
    - `chat()`: Direct chat interface
    - Supports batch processing and custom layer selection

### data/
- **base_dataset.py**: Abstract base class for datasets
  - `SurgicalPhaseDataset`: Defines common interface
  - Handles image loading and preprocessing
  - Extensible for new datasets

- **cholec_dataset.py**: Cholec80 dataset implementation
  - 7 surgical phases
  - Frame extraction and annotation loading

- **autolaparo_dataset.py**: AutoLaparo dataset implementation
  - 8 surgical phases
  - Compatible with Cholec80 interface

### utils/
- **transforms.py**: Image preprocessing
  - `build_transform()`: Standard InternVL3 transforms
  - `dynamic_preprocess()`: Dynamic image tiling
  - `preprocess_image()`: Complete preprocessing pipeline

- **helpers.py**: Utility functions
  - Frame indexing and sorting
  - Centroid-based prediction
  - Temporal smoothing
  - Result saving/loading

## Usage Patterns

### 1. Inference
```bash
python inference.py --dataset cholec --gpu 0
```

### 2. Adding a New Dataset
1. Create `data/your_dataset.py` inheriting from `SurgicalPhaseDataset`
2. Implement `_get_frame_list()` and `get_phase_names()`
3. Add configuration to `configs/config.py`
4. Update `data/__init__.py` with dataset factory

### 3. Custom Training
1. Modify `train.py` to implement your loss function
2. Update `SurgicalPhaseClassifier` forward pass
3. Implement training loop

### 4. Extending the Model
```python
from models import InternVL3Wrapper

# Custom layer extraction
model = InternVL3Wrapper(
    model_path="OpenGVLab/InternVL3-8B",
    layer_idx=15  # Different layer
)

# Batch processing
embeddings = model.extract_embeddings_batch(
    pixel_values_list, 
    num_patches_lists
)
```

## Design Principles

1. **Modularity**: Each component has a single, clear responsibility
2. **Extensibility**: Easy to add new datasets, models, or loss functions
3. **Clarity**: Simple, readable code with comprehensive docstrings
4. **Flexibility**: Support for custom transforms, configurations, and workflows
5. **Best Practices**: Follows PyTorch and research code conventions

## Future Extensions

- [ ] Implement training loop with custom loss
- [ ] Add multi-GPU support
- [ ] Add evaluation metrics (accuracy, F1, confusion matrix)
- [ ] Support for online/streaming inference
- [ ] Model checkpointing and resuming
- [ ] Weights & Biases integration
- [ ] Cross-validation utilities
- [ ] Data augmentation strategies

## Key Features

✅ Clean argument parsing with argparse
✅ Dataset factory pattern for easy extension
✅ Modular architecture for research flexibility
✅ Comprehensive documentation
✅ Ready for GitHub publication
✅ Type hints and docstrings throughout
✅ Follows PyTorch best practices
