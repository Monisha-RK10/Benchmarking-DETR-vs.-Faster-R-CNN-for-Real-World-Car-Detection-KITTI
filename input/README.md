# input/

This folder contains scripts and metadata to preprocess and organize the KITTI dataset for both Faster R-CNN and DETR pipelines.

---

## Files

### `split_kitti_dataset.py` - for Faster R-CNN
Splits the KITTI dataset into training and validation sets.

- Accepts image directory path.
- Generates `train.txt` and `val.txt` based on a specified split ratio and random seed.

**Run Example:**
```bash
python input/split_kitti_dataset.py \
    --image_dir faster-rcnn/train/images \
    --train_file input/train.txt \
    --val_file input/val.txt \
    --train_ratio 0.8
```
### `train.txt & val.txt` - for Faster R-CNN & DETR
- Lists of image base names (no extensions).
- Used to filter dataset entries during training/validation.
- Referenced in both Faster R-CNN (FilteredKITTIDataset) and DETR (generate_coco_json.py).

### `generate_coco_json.py` - for DETR
- Generates COCO-format JSON annotation files from the KITTI dataset for DETR.
- Requires KITTI labels and image splits from train.txt/val.txt.
- Outputs annotations.json compatible with HuggingFace DETR.
- Generates train_annotations.coco.json and val_annotations.coco.json.

### `train_annotations.coco.json & val_annotations.coco.json` - for DETR
- COCO-format JSON annotation for training dataset
- COCO-format JSON annotation for validation dataset
