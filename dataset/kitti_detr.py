import os
import torchvision
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

# Step 2 for DETR: To create train & val dataset.
# CocoDetection (parent): Loads image file and raw annotations from JSON.
# DetrImageProcessor: Preprocesses image (resizes, normalizes), converts boxes and labels into DETR format.
# Our custom class: Bridges both, uses parent to load raw data, processor to prep for model.

# CocoDetection (parent), apply image processor
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path) # Parent class' constructor
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx) # Parent class' getitem
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target

# Step 3 for DETR: Update the collate function.
# Recieve the batch, extract pixel values and labels.
# Collect images in batch, find the largest H, W in the batch, pad the images in the image to that size.
# Creates a pixel mask: 1 = valid pixel, 0 = padding (used in attention masking inside DETR).

def collate_fn(batch):
    pixel_values = [item[0] for item in batch] # Each item in batch is a tuple: (pixel_values, labels)
    labels = [item[1] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
