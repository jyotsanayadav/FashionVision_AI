import json
import os
import csv

import numpy as np
import torch

from torchvision.models.detection import maskrcnn_resnet50_fpn
from modanet_dataset import ModaNetDataset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ANNOTATION_FILE = (
    "dataset/modanet-annotations/annotations/"
    "modanet2018_instances_train.json"
)

IMAGE_DIR = "dataset/images"
SPLIT_FILE = "results/modanet_split.json"
MODEL_PATH = "models/modanet_maskrcnn_pilot.pth"

OUTPUT_FILE = "results/modanet_evaluation.csv"

NUM_CLASSES = 14
NUM_IMAGES = 100
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.01

DEVICE = torch.device("cpu")


# --------------------------------------------------
# IOU CALCULATION
# --------------------------------------------------

def box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def mask_iou(mask1, mask2):

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union if union > 0 else 0


# --------------------------------------------------
# AVERAGE PRECISION
# --------------------------------------------------

def calculate_ap(records, total_gt):

    if total_gt == 0:
        return None

    if not records:
        return 0.0

    records = sorted(
        records,
        key=lambda x: x["score"],
        reverse=True
    )

    tp = np.array([r["tp"] for r in records])
    fp = np.array([r["fp"] for r in records])

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-10)

    # Precision-recall interpolation
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(
            precision[i],
            precision[i + 1]
        )

    indices = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum(
        (recall[indices + 1] - recall[indices])
        * precision[indices + 1]
    )

    return float(ap)


# --------------------------------------------------
# MATCH PREDICTIONS WITH GROUND TRUTH
# --------------------------------------------------

def match_predictions(predictions, gt_boxes, gt_masks,
                      gt_labels, metric):

    records = []
    total_gt = {}

    for category in range(1, NUM_CLASSES):

        total_gt[category] = int(
            (gt_labels == category).sum()
        )

        category_predictions = [
            p for p in predictions
            if p["label"] == category
        ]

        category_predictions.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        matched = set()

        for prediction in category_predictions:

            best_iou = 0
            best_index = -1

            for i in range(len(gt_labels)):

                if gt_labels[i] != category or i in matched:
                    continue

                if metric == "box":
                    iou = box_iou(
                        prediction["box"],
                        gt_boxes[i]
                    )

                else:
                    iou = mask_iou(
                        prediction["mask"],
                        gt_masks[i]
                    )

                if iou > best_iou:
                    best_iou = iou
                    best_index = i

            is_tp = best_iou >= IOU_THRESHOLD

            if is_tp:
                matched.add(best_index)

            records.append({
                "category": category,
                "score": prediction["score"],
                "tp": int(is_tp),
                "fp": int(not is_tp)
            })

    return records, total_gt


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def evaluate():

    print("\nFASHIONVISION AI - MASK R-CNN EVALUATION")
    print("----------------------------------------")

    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    val_ids = set(split["val_ids"])

    dataset = ModaNetDataset(
        ANNOTATION_FILE,
        IMAGE_DIR,
        val_ids
    )

    num_images = min(NUM_IMAGES, len(dataset))

    print("Validation images:", len(dataset))
    print("Images selected for evaluation:", num_images)

    # Load model
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES
    )

    model.transform.min_size = (320,)
    model.transform.max_size = 320

    model.rpn._pre_nms_top_n = {
        "training": 500,
        "testing": 250
    }

    model.rpn._post_nms_top_n = {
        "training": 100,
        "testing": 50
    }

    model.roi_heads.detections_per_img = 50

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")
    print("\nStarting evaluation...\n")

    box_records = []
    mask_records = []

    box_gt_total = {}
    mask_gt_total = {}

    with torch.inference_mode():

        for image_index in range(num_images):

            image, target = dataset[image_index]

            prediction = model([
                image.to(DEVICE)
            ])[0]

            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()
            masks = prediction["masks"].cpu().numpy()

            predictions = []

            for box, label, score, mask in zip(
                boxes, labels, scores, masks
            ):

                predictions.append({
                    "box": box,
                    "label": int(label),
                    "score": float(score),
                    "mask": mask[0] >= 0.5
                })

            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()
            gt_masks = target["masks"].numpy().astype(bool)

            # Evaluate bounding boxes
            b_records, b_gt = match_predictions(
                predictions,
                gt_boxes,
                gt_masks,
                gt_labels,
                "box"
            )

            # Evaluate segmentation masks
            m_records, m_gt = match_predictions(
                predictions,
                gt_boxes,
                gt_masks,
                gt_labels,
                "mask"
            )

            box_records.extend(b_records)
            mask_records.extend(m_records)

            for category in range(1, NUM_CLASSES):

                box_gt_total[category] = (
                    box_gt_total.get(category, 0)
                    + b_gt[category]
                )

                mask_gt_total[category] = (
                    mask_gt_total.get(category, 0)
                    + m_gt[category]
                )

            if (image_index + 1) % 10 == 0:
                print(
                    f"Evaluated {image_index + 1}/{num_images} images"
                )

    # --------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------

    box_results = {}
    mask_results = {}

    for category in range(1, NUM_CLASSES):

        box_results[category] = calculate_ap(
            [
                r for r in box_records
                if r["category"] == category
            ],
            box_gt_total.get(category, 0)
        )

        mask_results[category] = calculate_ap(
            [
                r for r in mask_records
                if r["category"] == category
            ],
            mask_gt_total.get(category, 0)
        )

    valid_box_aps = [
        v for v in box_results.values()
        if v is not None
    ]

    valid_mask_aps = [
        v for v in mask_results.values()
        if v is not None
    ]

    box_map = np.mean(valid_box_aps) if valid_box_aps else 0
    mask_map = np.mean(valid_mask_aps) if valid_mask_aps else 0

    # Precision, recall and F1 at confidence threshold
    selected_box = [
        r for r in box_records
        if r["score"] >= CONFIDENCE_THRESHOLD
    ]

    tp = sum(r["tp"] for r in selected_box)
    fp = sum(r["fp"] for r in selected_box)

    total_gt = sum(box_gt_total.values())
    fn = total_gt - tp

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / total_gt if total_gt else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\nEVALUATION RESULTS")
    print("------------------")

    print(f"Images evaluated: {num_images}")
    print(f"Box mAP@0.5: {box_map:.4f}")
    print(f"Mask mAP@0.5: {mask_map:.4f}")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nCATEGORY-WISE RESULTS")
    print("---------------------")

    print(
        f"{'Category':<15}"
        f"{'Box AP':>12}"
        f"{'Mask AP':>12}"
    )

    rows = []

    for category in range(1, NUM_CLASSES):

        name = dataset.categories.get(
            category,
            "Unknown"
        )

        box_ap = box_results[category]
        mask_ap = mask_results[category]

        box_value = box_ap if box_ap is not None else 0
        mask_value = mask_ap if mask_ap is not None else 0

        print(
            f"{name:<15}"
            f"{box_value:>12.4f}"
            f"{mask_value:>12.4f}"
        )

        rows.append({
            "Category": name,
            "Box AP": box_value,
            "Mask AP": mask_value
        })

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["Category", "Box AP", "Mask AP"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nEvaluation saved:", OUTPUT_FILE)
    print("Evaluation completed!")


if __name__ == "__main__":
    evaluate()