### input/

This folder includes:

- `split_kitti_dataset.py`: Script to split the KITTI dataset into training and validation sets. It accepts parameters like image folder, train/val output paths, train split ratio, and random seed.
  
  **Run Example:**
  ```bash
  python input/split_kitti_dataset.py \
      --image_dir faster r-cnn/train/images \
      --train_file input/train.txt \
      --val_file input/val.txt \
      --train_ratio 0.8

- `train.txt & val.txt`: Text files listing image base names (without extension) used for training and validation splits, respectively.

- `generate_coco_json.py`: Script to generate train and val annotation COCO JSON for DETR. It requires train.txt and val.txt.
