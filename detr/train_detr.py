# Step 5 for DETR: Training

import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Dataset class and collate_fn imported from datasets/kitti_detr.py
from dataset.kitti_detr import KITTIDatasetDETR, collate_fn

# Model wrapper imported from detr/model_wrapper_detr.py
from model_wrapper_detr import Detr

# Step 6 for DETR: Reproducibility
def seed_everything(seed=42):
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# Step 7 for DETR: Define paths and config
CHECKPOINT = 'facebook/detr-resnet-50'
ANNOTATION_FILE_NAME = "_annotations.coco.json"
DATASET_LOCATION = "/content/drive/MyDrive/DETR/coco_update"
TRAIN_DIRECTORY = os.path.join(DATASET_LOCATION, "train")
VAL_DIRECTORY = os.path.join(DATASET_LOCATION, "valid")

# Step 8 for DETR: Load processor
# Image processor handles set of utilities for:
# a) Preprocessing such as image resizing, normalization, padding, conversion to tensors, and
# b) Post-processeing such as model outputs, turning raw logits into actual bounding boxes, labels, and scores.
# Set conf & iou threshold for inference/post processing
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# Step 9 for DETR: Load datasets and dataloaders
TRAIN_DATASET = KITTIDatasetDETR(image_directory_path=TRAIN_DIRECTORY,
                                 image_processor=image_processor, train=True)
VAL_DATASET = KITTIDatasetDETR(image_directory_path=VAL_DIRECTORY,
                               image_processor=image_processor, train=False)

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,
                              collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET,
                            collate_fn=collate_fn, batch_size=4)

# Step 10 for DETR: TensorBoard logger
logger = TensorBoardLogger("lightning_logs", name="detr")

# Step 11 for DETR: Initialize PyTorch Lightning model wrapper
lightning_model = Detr(
    lr=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4,
    train_dataloader=TRAIN_DATALOADER,
    val_dataloader=VAL_DATALOADER
)

# Step 12 for DETR: ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validation/loss",
    mode="min",
    dirpath="checkpoints/",
    filename="best-checkpoint",
    save_weights_only=True
)

# Step 13 for DETR: Trainer setup
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    devices=1,
    accelerator="gpu",
    max_epochs=40,
    gradient_clip_val=0.1,                                                                # Prevents exploding gradients: If gradients become too large (e.g. > 0.1), they get clipped
    accumulate_grad_batches=8,                                                            # Training acts like trained with batch size = number of batches * 8 
    log_every_n_steps=5
)

# Step 14 for DETR: Train the model
trainer.fit(lightning_model)

# Step 15 for DETR: Save the model and processor
MODEL_PATH_40_updated = "/content/drive/MyDrive/DETR/custom-model_40epochs_updated"
lightning_model.model.save_pretrained(MODEL_PATH_40_updated)                             # Full HuggingFace model + config (saves config, weight, tokenizer)
image_processor.save_pretrained(MODEL_PATH_40_updated)

# Optional: Save weights separately
torch.save(lightning_model.state_dict(), "lightning_detr_weights.pt")
