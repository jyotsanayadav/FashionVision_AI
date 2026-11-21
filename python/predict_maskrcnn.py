import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
OUTPUT_PATH = "results/modanet_maskrcnn_prediction.png"

NUM_CLASSES = 14
DEVICE = torch.device("cpu")
CONFIDENCE_THRESHOLD = 0.01


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

val_ids = set(split["val_ids"])

dataset = ModaNetDataset(
    ANNOTATION_FILE,
    IMAGE_DIR,
    val_ids
)

image, target = dataset[0]

print("Validation images:", len(dataset))
print("Selected image shape:", image.shape)


# --------------------------------------------------
# LOAD MASK R-CNN
# --------------------------------------------------

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


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

with torch.inference_mode():

    prediction = model([
        image.to(DEVICE)
    ])[0]

boxes = prediction["boxes"].cpu().numpy()
labels = prediction["labels"].cpu().numpy()
scores = prediction["scores"].cpu().numpy()
masks = prediction["masks"].cpu().numpy()

categories = dataset.categories

print("Maximum confidence:", scores.max() if len(scores) else 0)
print("Raw predictions:", len(scores))
# Keep predictions above confidence threshold.
keep = scores >= CONFIDENCE_THRESHOLD

boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]
masks = masks[keep]

print("\nPREDICTION RESULTS")
print("------------------")
print("Detected objects:", len(boxes))

for label, score in zip(labels, scores):

    category = categories.get(int(label), "Unknown")

    print(
        f"{category}: "
        f"Confidence = {score:.3f}"
    )


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

image_np = image.permute(1, 2, 0).numpy()

fig, ax = plt.subplots(figsize=(10, 12))

ax.imshow(image_np)

for box, label, score, mask in zip(
    boxes, labels, scores, masks
):

    x1, y1, x2, y2 = box

    # Draw segmentation boundary.
    binary_mask = mask[0] >= 0.5

    if binary_mask.any():
        ax.contour(
            binary_mask,
            levels=[0.5],
            linewidths=1.5
        )

    # Draw bounding box.
    rectangle = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="lime",
        facecolor="none"
    )

    ax.add_patch(rectangle)

    category = categories.get(int(label), "Unknown")

    ax.text(
        x1,
        max(0, y1 - 5),
        f"{category} ({score:.2f})",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="green", alpha=0.8)
    )

ax.set_title("FashionVision AI - Mask R-CNN Prediction")
ax.axis("off")

os.makedirs("results", exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    bbox_inches="tight",
    dpi=150
)

plt.close()

print("\nVisualization saved:", OUTPUT_PATH)
print("Prediction completed!")