### detr/

This folder contains code for training DETR using PyTorch Lightning.

- `model_wrapper.py`: Wraps Hugging Face's DETR model into a `pl.LightningModule`. Includes:
  - `KITTIDatasetDETR` class for loading data with `DetrImageProcessor`
  - `collate_fn` for dynamic padding and batching
  - Supports differential learning rates: backbone (ResNet) vs. transformer
  - Training and validation step logging, optimizer setup

