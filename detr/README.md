# detr/

This folder contains code for training and evaluating DETR (DEtection TRansformer) using PyTorch Lightning and Hugging Face's `transformers` library.

## Files

### `model_wrapper_detr.py`

Wraps Hugging Face's DETR model into a `pl.LightningModule`. Includes:
- Setting pretrained model, number of labels etc.
- Differential learning rates: separate LR for backbone (ResNet) and transformer
- Training and validation step logging
- Exposing train and val dataloader

### `train_detr.py`

Script to train DETR on the custom KITTI car-only dataset.
- Creating dataset, dataloader, logging TensorBoard
- Initializes PyTorch Lightning `Trainer` with logging and checkpointing
- Supports resuming from checkpoints and modifying training hyperparameters

### `inference_visualize_&_evaluate_detr.py`

Script for inference, visualization, and COCO-style evaluation.
- Runs inference on the validation set using a confidence threshold
- Visualizes both ground truth and predicted bounding boxes with `supervision`
- Saves results and zips them for easy sharing
- Computes mAP and IoU metrics using `CocoEvaluator`
