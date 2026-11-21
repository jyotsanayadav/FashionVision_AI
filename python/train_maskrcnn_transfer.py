import json
import os
import csv
import time

import torch
from torch.utils.data import DataLoader

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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

MODEL_PATH = "models/modanet_maskrcnn_transfer.pth"
HISTORY_PATH = "results/modanet_maskrcnn_transfer.csv"

NUM_CLASSES = 14

TRAIN_SAMPLES = 200
EPOCHS = 3
START_EPOCH = 3

BATCH_SIZE = 1

DEVICE = torch.device("cpu")


# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


def load_datasets():

    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    train_ids = set(split["train_ids"][:TRAIN_SAMPLES])

    dataset = ModaNetDataset(
        ANNOTATION_FILE,
        IMAGE_DIR,
        train_ids
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    return loader


# --------------------------------------------------
# TRANSFER LEARNING MODEL
# --------------------------------------------------

def create_model():

    print("\nLoading pretrained COCO Mask R-CNN weights...")

    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    print("Pretrained model loaded!")

    # Replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    # Replace segmentation mask head
    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        NUM_CLASSES
    )

    # Freeze pretrained backbone initially
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    # Reduce image size for CPU training
    model.transform.min_size = (320,)
    model.transform.max_size = 320

    # Reduce region proposals
    model.rpn._pre_nms_top_n = {
        "training": 500,
        "testing": 250
    }

    model.rpn._post_nms_top_n = {
        "training": 100,
        "testing": 50
    }

    model.roi_heads.detections_per_img = 50

    return model


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train():

    torch.set_num_threads(4)

    print("\nFASHIONVISION AI - MASK R-CNN TRANSFER LEARNING")
    print("-----------------------------------------------")
    print("Device:", DEVICE)
    print("Training samples:", TRAIN_SAMPLES)
    print("Epochs:", EPOCHS)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader = load_datasets()

    print("\nDataset loaded successfully!")
    print("Images:", len(train_loader.dataset))

    model = create_model()
    model.to(DEVICE)

    if START_EPOCH > 1:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    print("Resumed from saved checkpoint!")

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=0.0005,
        momentum=0.9,
        weight_decay=0.0005
    )

    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = list(csv.DictReader(f))

    for epoch in range(START_EPOCH - 1, EPOCHS):

        model.train()

        # Keep the frozen backbone in evaluation mode
        model.backbone.eval()

        total_loss = 0.0
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for step, (images, targets) in enumerate(train_loader):

            images = [
                image.to(DEVICE)
                for image in images
            ]

            targets = [
                {
                    key: value.to(DEVICE)
                    for key, value in target.items()
                }
                for target in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()

            optimizer.step()

            total_loss += losses.item()

            print(
                f"Step {step + 1}/{len(train_loader)} "
                f"| Loss: {losses.item():.4f}"
            )

        average_loss = total_loss / len(train_loader)

        elapsed = time.time() - start_time

        history.append({
            "Epoch": epoch + 1,
            "Training Loss": average_loss,
            "Time (seconds)": elapsed
        })

        print("\nAverage Training Loss:", round(average_loss, 4))
        print("Time:", round(elapsed, 2), "seconds")

        # Save checkpoint after every epoch
        torch.save(model.state_dict(), MODEL_PATH)

        with open(HISTORY_PATH, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Epoch",
                    "Training Loss",
                    "Time (seconds)"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

    print("\nTRANSFER LEARNING COMPLETED!")
    print("Model saved:", MODEL_PATH)
    print("History saved:", HISTORY_PATH)


if __name__ == "__main__":
    train()