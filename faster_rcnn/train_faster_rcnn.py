# Step 5 for Faster R-CNN: Create dataset & dataloader for train & val

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Dataset class and collate_fn imported from datasets/kitti.py
from dataset.kitti import FilteredKITTIDataset, collate_fn

train_dataset = FilteredKITTIDataset(
    '/content/drive/MyDrive/faster r-cnn/train/images',
    '/content/drive/MyDrive/faster r-cnn/train/labels',
    'input/train.txt',
    #transforms = None
)

val_dataset = FilteredKITTIDataset(
    '/content/drive/MyDrive/faster r-cnn/train/images',
    '/content/drive/MyDrive/faster r-cnn/train/labels',
    'input/val.txt'
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Step 6 for Faster R-CNN: Load model, set number of classes, & change the layer accordingly

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)                                                     # Finetuning (handles backbone, RPN, ROI Heads (except classifier head, replaced), optimizer, etc. internally)
num_classes = 2                                                                                      # 1 class ('Car') + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)                           # SGD: Backbone (e.g., ResNet) is well-behaved with SGD, weight decay: L2 regularization (prevents overfitting)

# Learning rate scheduler
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)                                            # Reduces the LR every few epochs no matter what. Epochs 0–9 -> LR = 0.005, Epochs 10–19 -> LR = 0.0005

# Training loop with best model saving
best_val_loss = float('inf')
best_model_path = "fasterrcnn_best1.pth"
num_epochs = 40

for epoch in range(num_epochs):
    # Training
    model.train()                                                                                    # Input: images + targets, Output: loss_dict for torchvision
    train_loss = 0.0
    # Initialize at start of epoch
    loss_comp_sum = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }
    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        for k in loss_dict:
            loss_comp_sum[k] += loss_dict[k].item()
        #for k, v in loss_dict.items():
          #  print(f"Epoch [{epoch+1}] - {k}: {v.item():.4f}")                                       # Losses like loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    # model.eval()                                                                                   # Input: images only (targets = None) Output: predictions 
    model.train()                                                                                    # NOTE: Needed to compute loss in torchvision models, switch back to train mode to compute loss
    val_loss = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    model.eval()                                                                                     # Return to eval mode after validation for safety and good practice. Not the best practice, switch modes immediately to have less batchnorm, dropout influence

    avg_val_loss = val_loss / len(val_loader)

    # Update the learning rate
    lr_scheduler.step()
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    # After epoch loop
    num_batches = len(train_loader)
    for k in loss_comp_sum:
        print(f"Epoch [{epoch+1}] - Avg {k}: {loss_comp_sum[k]/num_batches:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Optional: save model each epoch
    # torch.save(model.state_dict(), f'fasterrcnn_epoch{epoch+1}.pth')                               # Uncomment if you want per-epoch checkpoints
