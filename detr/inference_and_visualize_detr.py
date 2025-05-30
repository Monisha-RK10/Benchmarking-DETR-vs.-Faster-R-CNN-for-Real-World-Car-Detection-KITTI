import os
import cv2
import torch
import zipfile
import numpy as np
import supervision as sv
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

# Step 16 for DETR: Inference on all images from val dataset.

# ---- CONFIG ---- #
SAVE_DIR = '/content/DETR_inference_results_val_conf_0.9_updated'
ZIP_PATH = '/content/DETR_inference_results_val_conf_0.9_updated.zip'
THRESHOLD = 0.9

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)

# Utils
categories = VAL_DATASET.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}
box_annotator = sv.BoxAnnotator()

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

    # ---- GROUND TRUTH ----
    if len(annotations) > 0:
        detections_gt = sv.Detections.from_coco_annotations(coco_annotation=annotations)
        labels_gt = [f"{id2label[class_id]}" for _, _, class_id, _ in detections_gt]
        frame_gt = box_annotator.annotate(scene=image.copy(), detections=detections_gt, labels=labels_gt)
    else:
        print('No GT boxes')
        frame_gt = image.copy()  # No GT boxes

    # ---- PREDICTIONS ----
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=THRESHOLD,
            target_sizes=target_sizes
        )[0]

    # Handle empty detection results
    if len(results["scores"]) > 0:
        detections_pred = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.9)
        labels_pred = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_pred]
        frame_pred = box_annotator.annotate(scene=image.copy(), detections=detections_pred, labels=labels_pred)
    else:
        frame_pred = image.copy()  # No predictions

    # Save both GT and predictions
    filename_base = os.path.splitext(os.path.basename(img_info['file_name']))[0]
    cv2.imwrite(os.path.join(SAVE_DIR, f"{filename_base}_gt.jpg"), frame_gt)
    cv2.imwrite(os.path.join(SAVE_DIR, f"{filename_base}_pred.jpg"), frame_pred)

# ---- ZIP the results ---- #
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(SAVE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, SAVE_DIR)
            zipf.write(file_path, arcname)

print(f"\n Done! Results saved to: {SAVE_DIR}")
print(f" Zipped file ready at: {ZIP_PATH}")

# Step 17 for DETR: Evaluation via coco eval.

# Convert xywh to xyxy format.
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

# Prepare the result for coco evaluation.
def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
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
    return coco_results

# Step 18 for DETR: Extract validation metrics.
# Model predictions (raw outputs -> cxcywh, normalized).
# Post-processing to get final boxes (xyxy -> absolute pixel coords), scores, and labels.
# Get results (boxes, labels, scores in real image size).
# Prepare_for_coco_detection(results), convert_to_xywh().
# Get [{"image_id", "category_id", "bbox", "score"}, ...]
# Evaluation using CocoEvaluator (evaluator.update(...), evaluator.accumulate(), summarize())

evaluator = CocoEvaluator(coco_gt=VAL_DATASET.coco, iou_types=["bbox"])

print("Running evaluation...")

for idx, batch in enumerate(tqdm(VAL_DATALOADER)):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes) # cx, cy, w, h ∈ [0, 1]  # normalized to image size to xyxy absolute pixels for drawing boxes & computing IoU

    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
