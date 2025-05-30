import os
import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Dataset class and collate_fn imported from datasets/kitti_detr.py
from dataset.kitti_detr import KITTIDatasetDETR, collate_fn

# Step 6 for DETR: Load dataset & dataloader
# Set paths

ANNOTATION_FILE_NAME = "_annotations.coco.json"
DATASET_LOCATION = "/content/drive/MyDrive/DETR/coco_update"
TRAIN_DIRECTORY = os.path.join(DATASET_LOCATION, "train")
VAL_DIRECTORY = os.path.join(DATASET_LOCATION, "valid")

# Train & val dataset
TRAIN_DATASET = KITTIDatasetDETR(image_directory_path=TRAIN_DIRECTORY,image_processor=image_processor,train=True)
VAL_DATASET = KITTIDatasetDETR(image_directory_path=VAL_DIRECTORY,image_processor=image_processor,train=False)

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)

# Step 7 for DETR: Load model & image processor.
# Load both image processor & object detection model (they both must have the same model).
# Image processor handles set of utilities for:
# a) Preprocessing such as image resizing, normalization, padding, conversion to tensors, and
# b) Post-processeing such as model outputs, turning raw logits into actual bounding boxes, labels, and scores.

# Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'

# Processor
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

# Step 8 for DETR: Load tensor board for tracking the model's progress.

%load_ext tensorboard
%tensorboard --logdir lightning_logs/
logger = TensorBoardLogger("lightning_logs", name="detr") # Set up TensorBoard logging for PyTorch Lightning.

# Step 9 for DETR: Training & evaluation.
# Train the model while ensuring reproducibility, tracking loss, & setting hyperparameters.

# Reproducibility
def seed_everything(seed=42):
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# Define the model
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validation/loss",
    mode="min",
    dirpath="checkpoints/",
    filename="best-checkpoint",
    save_weights_only=True
)

# Trainer settings
MAX_EPOCHS = 40
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    devices=1,
    accelerator="gpu",
    max_epochs=MAX_EPOCHS,
    gradient_clip_val=0.1,
    accumulate_grad_batches=8,
    log_every_n_steps=5
)

# Train
trainer.fit(model)

# Step 10 for DETR: Save the model, processor, & weights.

MODEL_PATH_40_updated = "/content/drive/MyDrive/DETR/custom-model_40epochs_updated"
model.model.save_pretrained(MODEL_PATH_40_updated)
image_processor.save_pretrained(MODEL_PATH_40_updated)

# Optional, save the weights
torch.save(model.state_dict(), "lightning_detr_weights.pt")
