import json
import os
import csv
import time

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

#from python.modanet_dataset import ModaNetDataset
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
HISTORY_PATH = "results/modanet_maskrcnn_pilot.csv"

NUM_CLASSES = 14  # 13 clothing categories + background

TRAIN_SAMPLES = 200
EPOCHS = 3
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
# MODEL
# --------------------------------------------------

def create_model():

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES
    )

    # Reduce image size to make CPU training manageable.
    model.transform.min_size = (320,)
    model.transform.max_size = 320

    # Reduce region proposals for the pilot experiment.
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

    print("\nFASHIONVISION AI - MASK R-CNN")
    print("--------------------------------")
    print("Device:", DEVICE)
    print("Training samples:", TRAIN_SAMPLES)
    print("Epochs:", EPOCHS)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader = load_datasets()

    print("Dataset loaded successfully!")
    print("Images:", len(train_loader.dataset))

    model = create_model()
    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    history = []

    for epoch in range(EPOCHS):

        model.train()

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

    # Save model
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    # Save training history
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

    print("\nTRAINING COMPLETED!")
    print("Model saved:", MODEL_PATH)
    print("History saved:", HISTORY_PATH)


if __name__ == "__main__":
    train()