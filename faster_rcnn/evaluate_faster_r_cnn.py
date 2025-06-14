# Step 7 for Faster R-CNN: Evaluate faster r-cnn trained model on val loader

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from engine import evaluate                                                       # Torchvision's custom training/eval loop. It expects each target to contain boxes, labels, 'image_id': as a torch.tensor([idx]) or int
from coco_utils import convert_to_coco_api
from coco_eval import CocoEvaluator                                               # PyTorch’s built-in wrapper around COCOeval.
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os

# Dataset class and collate_fn imported from dataset/kitti.py
from dataset.kitti import FilteredKITTIDataset, collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_dataset = FilteredKITTIDataset(
    '/content/drive/MyDrive/faster r-cnn/train/images',
    '/content/drive/MyDrive/faster r-cnn/train/labels',
    'input/val.txt'
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)                  # Background + car
model.load_state_dict(torch.load("/content/fasterrcnn_best1.pth"))
model.to(device)

# Run evaluation
evaluate(model, val_loader, device=torch.device('cuda'))

# Visualization setup
save_dir = '/content/result_faster_rcnn_conf_0.9'
os.makedirs(save_dir, exist_ok=True)

def visualize_and_save(image, prediction, idx, threshold=0.9):
    image_np = image.permute(1, 2, 0).cpu().numpy()                               # Reorder [C, H, W] -> [H, W, C] for visualization
    plt.figure(figsize=(10,10))
    plt.imshow(image_np)

    boxes = prediction['boxes'].cpu().numpy()                                     # Use .detach().cpu().numpy() when inside training loops, logging predictions, or visualizing intermediate layers.
    scores = prediction['scores'].cpu().numpy()                                   # Use .cpu().numpy() when doing inference only and are sure there's no autograd graph.
    labels = prediction['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='r', linewidth=2))
            plt.gca().text(x1, y1, f'{label}: {score:.2f}', color='white',
                           bbox=dict(facecolor='blue', alpha=0.5))

    plt.axis('off')
    save_path = os.path.join(save_dir, f'pred_{idx}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Run inference and save visualizations for the val_loader
idx = 0
model.eval()
with torch.no_grad():
    for images, _ in val_loader:
        images_cuda = [img.to(device) for img in images]
        preds = model(images_cuda)
        for img, pred in zip(images, preds):
            visualize_and_save(img, pred, idx)
            idx += 1

print(f"Saved {idx} prediction images to {save_dir}")
