# Step 16 for DETR: Inference on all images from val dataset (visualize and save) using supervision.

import os
import cv2
import torch
import zipfile
import numpy as np
import supervision as sv                                                                                            # DETR's outputs are raw tensors, supervision helps convert, visualize, and apply NMS easily. Provides Conversion: .from_transformers(...), Visualization: .annotate(...), NMS: .with_nms(...)
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

# Config
SAVE_DIR = '/content/DETR_inference_results_val_conf_0.9_updated'
ZIP_PATH = '/content/DETR_inference_results_val_conf_0.9_updated.zip'
THRESHOLD = 0.9

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)

# Utils
categories = VAL_DATASET.coco.cats                                                                                  # KITTIDatasetDETR class internally wraps a COCO-style object (uses pycocotools.coco.COCO to parse and expose the dataset's annotations). Allows VAL_DATASET.coco.cats, VAL_DATASET.coco.getImgIds(), VAL_DATASET.coco.imgToAnns[image_id]
id2label = {k: v['name'] for k, v in categories.items()}                                                            # Similar to {0: 'car', 1: 'pedestrian', ...}, used for displaying class names on visualizations
box_annotator = sv.BoxAnnotator()                                                                                   # Used to draw bounding boxes with labels on images.

# Get all image IDs
image_ids = VAL_DATASET.coco.getImgIds()

# Inference loop
for image_id in tqdm(image_ids, desc="Processing Validation Set"):
    # Load image & annotations
    img_info = VAL_DATASET.coco.loadImgs(image_id)[0]
    annotations = VAL_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(VAL_DATASET.root, img_info['file_name'])
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not load image {image_path}")
        continue

    # Ground truth from val dataset
    if len(annotations) > 0:
        detections_gt = sv.Detections.from_coco_annotations(coco_annotation=annotations)                            # Converts GT annotations to supervision.Detections
        labels_gt = [f"{id2label[class_id]}" for _, _, class_id, _ in detections_gt]                                # Extracts class names for each box
        frame_gt = box_annotator.annotate(scene=image.copy(), detections=detections_gt, labels=labels_gt)           # Draws GT boxes on a copy of the image
    else:
        print('No GT boxes')
        frame_gt = image.copy()  # No GT boxes

    # Prediction from model inference
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)                                      # Prepares input tensors
        outputs = model(**inputs)                                                                                   # Python's "unpacking" operator for dictionaries, raw model output 

        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(                                                    # Post-processes prediction (converts raw logits to scores, boxes, and class IDs, applies conf thresh, rescales boxes to image size)
            outputs=outputs,
            threshold=THRESHOLD,
            target_sizes=target_sizes
        )[0]

    # Handle empty detection results
    if len(results["scores"]) > 0:
        detections_pred = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.9)     # Converts predictions to supervision.Detections format, applies NMS
        labels_pred = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_pred]  # Each detection is a tuple of (bbox, confidence, class_id, tracker_id)
        frame_pred = box_annotator.annotate(scene=image.copy(), detections=detections_pred, labels=labels_pred)     # Draws prediction boxes and class/confidence on a copy of the image
    else:
        frame_pred = image.copy()  # No predictions

    # Save both GT and predictions
    filename_base = os.path.splitext(os.path.basename(img_info['file_name']))[0]
    cv2.imwrite(os.path.join(SAVE_DIR, f"{filename_base}_gt.jpg"), frame_gt)
    cv2.imwrite(os.path.join(SAVE_DIR, f"{filename_base}_pred.jpg"), frame_pred)

# ZIP the results
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:                                                  # ZIP_DEFLATED: Tells zipfile to compress files
    for root, _, files in os.walk(SAVE_DIR):                                                                        # Recursively gets all files to be zipped
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, SAVE_DIR)
            zipf.write(file_path, arcname)

print(f"\n Done! Results saved to: {SAVE_DIR}")
print(f" Zipped file ready at: {ZIP_PATH}")

# Step 16 for DETR: Evaluation via coco eval.

# Convert xywh to xyxy format.
def convert_to_xywh(boxes):                                                                                         # Boxes: tensor of shape [N, 4], each row is [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = boxes.unbind(1)                                                                        # Unpacks across columns, returns four tensors: xmin, ymin, xmax, ymax
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)                                               # Stacks them back into a new [N, 4] tensor, now in [x, y, w, h] format

# Prepare the result for coco evaluation.
def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():                                                             # original_id: image ID (int), prediction: dict with boxes: tensor of shape [N, 4], scores: list or tensor [N], labels: list or tensor [N]
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()                                                                     # Converts tensor to Python list for JSON serialization
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(                                                                                        # Adds multiple items from another iterable
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results                                                                                             # [ {"image_id": 42, "category_id": 2, "bbox": [10, 20, 100, 100], "score": 0.92}, {"image_id": 42, "category_id": 1, "bbox": [200, 150, 50, 30], "score": 0.87},...]

# Step 17 for DETR: Extract validation metrics.
# Model predictions (raw outputs -> cxcywh, normalized).
# Post-processing to get final boxes (xyxy -> absolute pixel coords), scores, and labels.
# Get results (boxes, labels, scores in real image size).
# Prepare_for_coco_detection(results), convert_to_xywh().
# Get [{"image_id", "category_id", "bbox", "score"}, ...]
# Evaluation using CocoEvaluator (evaluator.update(...), evaluator.accumulate(), summarize())

evaluator = CocoEvaluator(coco_gt=VAL_DATASET.coco, iou_types=["bbox"])                                             # GT in COCO format, evaluating bounding boxes

print("Running evaluation...")

for idx, batch in enumerate(tqdm(VAL_DATALOADER)):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]                                     # boxes, labels, image_id, 'orig_size': Tensor([H, W]),.....

    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)                              # dim=0 -> new rows (i.e., stack along batch dimension)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)                # Converts raw outputs into boxes: [xmin, ymin, xmax, ymax], scores, labels. target_sizes help the processor scale predictions back to the original image size
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}                    # Create a dictionary to map each image_id to its predicitions. labels: GT metadata for each image in the batch, results: processed predictions (from post_process_object_detection), .item() converts it from a 0-D tensor to a Python int, e.g., 42
    predictions = prepare_for_coco_detection(predictions)                                                           # Example: first iteration for output target = labels[0] (has image_id) and output = results[0] (has boxes, scores, labels)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
