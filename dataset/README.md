# dataset/

This folder contains dataset loader implementations tailored for Faster R-CNN and DETR training using the KITTI dataset.

---

## `kitti.py` – For Faster R-CNN

### Classes:
- **`KITTIDataset`**: Loads full KITTI-style images and labels.
- **`FilteredKITTIDataset`**: Loads only a subset defined by `train.txt` or `val.txt`.

### Functions:
- **`collate_fn`**: Handles batch collation with variable-length bounding boxes per image.

Used in: `faster-rcnn/train_faster_rcnn.py`

---

## `kitti_detr.py` – For DETR

### Components:
- **`KITTIDatasetDETR`**:
  - Inherits from `torchvision.datasets.CocoDetection`.
  - Loads images + raw annotations.
  - Uses HuggingFace `DetrImageProcessor` for resizing, normalization, and box formatting.

- **`collate_fn_detr(batch, processor)`**:
  - Pads all images in a batch to the largest height and width.
  - Creates `pixel_mask` for attention masking in DETR.

Used in: `detr/train_detr.py`

---



