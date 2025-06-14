# Step 1 for Faster R-CNN: Split train/val (80/20) randomly & save them (train.txt, val.txt) for reproducibility
# Using .txt for labels from KITTI dataset, no pycocoo tools applied.

import os
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def split_dataset(image_dir, train_path='train.txt', val_path='val.txt', train_ratio=0.8, seed=42):
    valid_exts = ('.png', '.jpg', '.jpeg')
    all_images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])

    if len(all_images) == 0:
        raise ValueError(f"No image files found in {image_dir}")

    random.seed(seed)
    random.shuffle(all_images)

    split_idx = int(train_ratio * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    assert set(train_images).isdisjoint(set(val_images)), "Train/Val sets overlap!"

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_images) + '\n')
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_images) + '\n')

    logging.info(f"Dataset split completed. {len(train_images)} train, {len(val_images)} val images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split KITTI dataset into train/val text files.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--train_file', type=str, default='train.txt', help='Output train file')
    parser.add_argument('--val_file', type=str, default='val.txt', help='Output val file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    split_dataset(args.image_dir, args.train_file, args.val_file, args.train_ratio, args.seed)
