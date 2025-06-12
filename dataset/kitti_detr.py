import os
import torchvision
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

# Step 2 for DETR: To create train & val dataset.
# CocoDetection (parent): Loads image file and raw annotations from JSON.
# DetrImageProcessor: Preprocesses image (resizes, normalizes), converts boxes [cx, cy, w, h] and labels (integer tensors) into DETR format. Returns preprocessed image tensor & List of dictionaries (one per image)
# KITTIDatasetDETR custom class: Bridges both, uses parent to load raw data, processor to prep for model.

# Set paths
ANNOTATION_FILE_NAME = "_annotations.coco.json"

# Custom dataset for DETR using COCO-format annotations
class KITTIDatasetDETR(torchvision.datasets.CocoDetection):                                             # Custom class: KITTIDatasetDETR, parent class: torchvision.datasets.CocoDetection i.e., extending CocoDetection (reads COCO-style image + annotations)
    def __init__(self, image_directory_path: str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(KITTIDatasetDETR, self).__init__(image_directory_path, annotation_file_path)              # Calling super() once per method: parent class' constructor to set up base functionality (image paths, annotation loading)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(KITTIDatasetDETR, self).__getitem__(idx)                            # Calling super() once per method: parent class' getitem to reuse the logic that reads an image and its annotations, getitem(idx) automatically gives the image and the annotations linked to it by image_id.
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}                                # Not changing the annotations' content, just wrapping them in a dictionary
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")    
        pixel_values = encoding["pixel_values"].squeeze()                                               # To avoid unnecessary extra dimensions during DataLoader batching
        target = encoding["labels"][0]                                                                  # List of dicts, one per image. Only one image (not a batch) is given, so the first and only item with [0] is fetched.
        return pixel_values, target

# Step 3 for DETR: Update the collate function.
# Recieve the batch, extract pixel values and labels.
# Collect images in batch, find the largest H, W in the batch, pad the images in the batch to that size.
# Creates a pixel mask: 1 = valid pixel, 0 = padding (used in attention masking inside DETR).

def collate_fn(batch, image_processor: DetrImageProcessor):
    
    pixel_values = [item[0] for item in batch]                                                          # Each item in batch is a tuple: (pixel_values, labels) where batch = [(pv1, tgt1), (pv2, tgt2), (pv3, tgt3), (pv4, tgt4)]
    labels = [item[1] for item in batch]

    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    return {
        'pixel_values': encoding['pixel_values'],                                                        # Names must be exact, because DETR expects them this way internally 
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
