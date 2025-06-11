import os
import json
from shutil import copy2
from PIL import Image

# Step 1 for DETR: Generate json coco format.
# Extract train & val images from train & val .txt.
# Send them to the coco function to extract image, annotation, & categories in json format.
# For annotation, parse the label to extract boxes (x, y, w, h), area, category, iscrowd to match with coco format.
# Arrange the images & labels in the folder structure acceptable by DETR, i.e., folder with subfolders train & valid with corresonding images & json.

# Paths
base_path = "/content/drive/MyDrive/DETR"
image_path = os.path.join(base_path, "train", "images")
label_path = os.path.join(base_path, "train", "labels")
output_base = os.path.join(base_path, "coco_update")

train_txt_path = "/content/train.txt"
val_txt_path = "/content/val.txt"

# Output folders
output_train = os.path.join(output_base, "train")
output_valid = os.path.join(output_base, "valid")
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_valid, exist_ok=True)

# COCO-style category list (class_id = 1 for COCO, but DETR internally shifts this to class_id=0)
categories = [{"id": 0, "name": "Car"}]                                           # Ensure that this category match with id2label & label2id set in the model

def load_split(txt_path): # Train.txt, val.txt
    with open(txt_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def parse_label_file(label_file):
    annotations = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            if parts[0].lower() != "car":                                         # Handle 'car', 'Car', etc.
                continue
            try:
                x_min = float(parts[4])
                y_min = float(parts[5])
                x_max = float(parts[6])
                y_max = float(parts[7])
                width = x_max - x_min
                height = y_max - y_min
                annotations.append({
                    "bbox": [x_min, y_min, width, height],                        # COCO format
                    "area": width * height,
                    "category_id": 0,                                             # COCO-style ID (DETR will convert to 0 internally)
                    "iscrowd": 0,
                })
            except Exception as e:
                print(f"Skipping invalid line in {label_file}: {line}")
    return annotations

def generate_coco_json(image_ids, output_img_folder, json_output_path):
    images = []
    annotations = []
    ann_id = 1
    for i, image_file in enumerate(image_ids):
        img_name = image_file
        label_name = image_file.replace(".png", ".txt")
        img_src = os.path.join(image_path, img_name)
        lbl_src = os.path.join(label_path, label_name)

        if not os.path.exists(img_src):
            print(f"Image not found: {img_src}")
            continue

        # Copy image to output
        copy2(img_src, os.path.join(output_img_folder, img_name))

        # Open the image to get actual size
        with Image.open(img_src) as img:
            width, height = img.size

        # Image metadata
        images.append({
            "id": i,
            "file_name": img_name,
            "height": height,
            "width": width,
        })

        # Parse and add annotations
        anns = []
        if os.path.exists(lbl_src):
            anns = parse_label_file(lbl_src)

        for ann in anns:
            ann["image_id"] = i
            ann["id"] = ann_id
            ann_id += 1
            annotations.append(ann)

    # Build JSON
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(json_output_path, "w") as f:
        json.dump(coco_dict, f)

    print(f"Saved: {json_output_path}")

# Load image splits
train_ids = load_split(train_txt_path)
val_ids = load_split(val_txt_path)

# Generate COCO JSONs
generate_coco_json(train_ids, output_train, os.path.join(output_train, "_annotations.coco.json"))
generate_coco_json(val_ids, output_valid, os.path.join(output_valid, "_annotations.coco.json"))

print("COCO-style JSONs created and images copied.")
