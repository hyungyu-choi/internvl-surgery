# InternVL3 Surgical Phase Recognition

Clean, modular implementation for surgical phase recognition using InternVL3.

## Project Structure

```
internvl_surgery/
├── models/
│   ├── __init__.py
│   └── internvl_wrapper.py      # InternVL3 model wrapper
├── data/
│   ├── __init__.py
│   ├── base_dataset.py          # Base dataset class
│   ├── cholec_dataset.py        # Cholec80 dataset
│   └── autolaparo_dataset.py    # AutoLaparo dataset
├── utils/
│   ├── __init__.py
│   ├── transforms.py            # Image preprocessing
│   └── helpers.py               # Helper functions
├── configs/
│   ├── __init__.py
│   └── config.py                # Configuration management
├── train.py                      # Training script (for future finetuning)
├── inference.py                  # Inference script
└── README.md
```

## Installation

```bash
pip install torch torchvision transformers pillow tqdm numpy
```

## Quick Start

### Inference
```bash
python inference.py \
    --dataset cholec \
    --model_path OpenGVLab/InternVL3-8B \
    --gpu 0
```

### Training (Coming Soon)
```bash
python train.py \
    --dataset cholec \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4
```

## Supported Datasets

- **Cholec80**: Cholecystectomy surgical videos
- **AutoLaparo**: Laparoscopic surgical videos

## Citation

If you use this code, please cite InternVL3:
```bibtex
@article{internvl3,
  title={InternVL3: Scaling Up Vision-Language Models},
  author={...},
  year={2024}
}
```
