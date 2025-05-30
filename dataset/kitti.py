# Step 2 for Faster R-CNN: Create a custom dataset class.
# This step does the following:
# Open image in PIL format, convert to RGB, and apply basic transform/toTensor.
# Extract bbox, labels based on the class map.
# If an image has no car, handle it gracefully.
# Return image & target.

import os
import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class KITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None, verbose=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.verbose = verbose
        self.image_files = sorted(os.listdir(image_dir))

    def __getitem__(self, idx): # Retrieves image-label pair at index
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img) if self.transforms else ToTensor()(img)

        # Try loading annotations
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            if self.verbose:
                logging.warning(f"Label file not found for image: {img_name}. Skipping annotations.")
        else:
            with open(label_path) as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) < 8: # class, truncation, occlusion, alpha, x1, y1, x2, y2 (bbox)
                        continue  # Skip malformed lines
                    cls = fields[0].lower()
                    if cls == "dontcare":
                        continue
                    if cls != "car":
                        continue
                    xmin, ymin, xmax, ymax = map(float, fields[4:8])
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # class ID for 'Car'

        # Handle images with no valid boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': idx, # torch.tensor([idx]),tensor needed for training,.item() is called internally, idx for inference (only needs image)
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), # for validation
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64), # for validation

        }
        return img, target

    def __len__(self):
        return len(self.image_files)

# Step 3 for Faster R-CNN: Update collate function
# Faster R-CNN expects lists of targets, not a stacked tensor (PyTorch defaut).

def collate_fn(batch):
    return tuple(zip(*batch)) # tuple of images ->(img1, img2, ...),  tuple of targets -> (target1, target2, ...)

# Step 4 for Faster R-CNN: Subclass FilteredKITTIDataset of KITTIDataset
# This filters images based on image filenames in train.txt, val.txt (check input folder)

class FilteredKITTIDataset(KITTIDataset):
    def __init__(self, image_dir, label_dir, image_list_file, transforms=None):
        super().__init__(image_dir, label_dir, transforms)

        # Load selected image filenames from train.txt or val.txt
        with open(image_list_file) as f:
            selected = set(f.read().splitlines())

        # Filter image_files based on the selection
        self.image_files = [f for f in self.image_files if f in selected]

        # Safety check to catch mistakes early
        assert len(self.image_files) > 0, f"No matching images found in {image_list_file}"
