# detr/

This folder contains code for training and evaluating DETR (DEtection TRansformer) using PyTorch Lightning and Hugging Face's `transformers` library.

## Files

### `model_wrapper_detr.py`

Wraps Hugging Face's DETR model into a `pl.LightningModule`. Includes:
- `KITTIDatasetDETR` class for loading data using `DetrImageProcessor`
- `collate_fn` for dynamic padding and batching
- Differential learning rates: separate LR for backbone (ResNet) and transformer
- Training and validation step logging
- Optimizer and scheduler setup

### `train_detr.py`

Script to train DETR on the custom KITTI car-only dataset.
- Loads and splits the dataset using `train.txt` and `val.txt`
- Applies `DetrImageProcessor` for input formatting
- Initializes PyTorch Lightning `Trainer` with logging and checkpointing
- Supports resuming from checkpoints and modifying training hyperparameters

### `inference_visualize_&_evaluate_detr.py`

Script for inference, visualization, and COCO-style evaluation.
- Runs inference on the validation set using a confidence threshold
- Visualizes both ground truth and predicted bounding boxes with `supervision`
- Saves results and zips them for easy sharing
- Computes mAP and IoU metrics using `CocoEvaluator`
