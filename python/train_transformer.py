import json
import os
import csv
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------- CONFIGURATION ----------------

METADATA_FILE = "results/transformer_metadata.json"
SPLIT_FILE = "results/transformer_metadata_split.json"

MODEL_PATH = "models/fashion_transformer.pth"
HISTORY_PATH = "results/transformer_history.csv"

NUM_FEATURES = 6
NUM_CLASSES = 13

D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

MAX_OBJECTS = 30

DEVICE = torch.device(
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu"
)


# ---------------- DATASET ----------------

class FashionMetadataDataset(Dataset):

    def __init__(self, metadata, image_ids):
        self.samples = []

        for image_id in image_ids:
            objects = metadata[str(image_id)]["objects"]

            features = [
               obj["features"][:5] + [
                   math.log(max(obj["features"][5], 1e-6))
               ]
               for obj in objects
            ]
            labels = [obj["label"] - 1 for obj in objects]

            self.samples.append((features, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        features, labels = self.samples[index]

        # Limit extremely long sequences.
        features = features[:MAX_OBJECTS]
        labels = labels[:MAX_OBJECTS]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )


def collate_fn(batch):

    max_length = max(len(item[0]) for item in batch)

    batch_size = len(batch)

    features = torch.zeros(
        batch_size, max_length, NUM_FEATURES
    )

    labels = torch.full(
        (batch_size, max_length), -100, dtype=torch.long
    )

    padding_mask = torch.ones(
        batch_size, max_length, dtype=torch.bool
    )

    for i, (x, y) in enumerate(batch):
        length = len(x)

        features[i, :length] = x
        labels[i, :length] = y
        padding_mask[i, :length] = False

    return features, labels, padding_mask


# ---------------- TRANSFORMER MODEL ----------------

class FashionTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(NUM_FEATURES, D_MODEL)

        self.position_embedding = nn.Parameter(
            torch.randn(1, MAX_OBJECTS, D_MODEL) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.classifier = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x, padding_mask):

        x = self.embedding(x)

        x = x + self.position_embedding[:, :x.size(1), :]

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        return self.classifier(x)


# ---------------- EVALUATION ----------------

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels, padding_mask in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)

            outputs = model(features, padding_mask)

            loss = criterion(
                outputs.reshape(-1, NUM_CLASSES),
                labels.reshape(-1)
            )

            valid = labels != -100

            total_loss += loss.item()
            predictions = outputs.argmax(dim=-1)

            correct += (
                (predictions == labels) & valid
            ).sum().item()

            total += valid.sum().item()

    return (
        total_loss / max(len(loader), 1),
        correct / max(total, 1)
    )


# ---------------- TRAINING ----------------

def main():

    print("FASHIONVISION AI - TRANSFORMER METADATA MODEL")
    print("----------------------------------------------")
    print("Device:", DEVICE)

    print("\nLoading metadata...")

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    train_dataset = FashionMetadataDataset(
        metadata, splits["train"]
    )

    val_dataset = FashionMetadataDataset(
        metadata, splits["validation"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    model = FashionTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_accuracy = 0

    with open(HISTORY_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Train Accuracy",
            "Validation Loss",
            "Validation Accuracy",
            "Time"
        ])

        for epoch in range(EPOCHS):

            start_time = time.time()

            model.train()

            total_loss = 0
            correct = 0
            total = 0

            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            for step, (features, labels, padding_mask) in enumerate(train_loader):

                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                padding_mask = padding_mask.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(features, padding_mask)

                loss = criterion(
                    outputs.reshape(-1, NUM_CLASSES),
                    labels.reshape(-1)
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

                valid = labels != -100
                predictions = outputs.argmax(dim=-1)

                correct += (
                    (predictions == labels) & valid
                ).sum().item()

                total += valid.sum().item()

                if (step + 1) % 100 == 0:
                    print(
                        f"Step {step + 1}/{len(train_loader)} "
                        f"| Loss: {loss.item():.4f}"
                    )

            train_loss = total_loss / len(train_loader)
            train_accuracy = correct / max(total, 1)

            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion
            )

            elapsed = time.time() - start_time

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Time: {elapsed:.2f} seconds")

            writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                elapsed
            ])

            f.flush()

            if val_accuracy > best_accuracy:

                best_accuracy = val_accuracy

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "epoch": epoch + 1
                }, MODEL_PATH)

                print("Best model saved!")

    print("\nTRAINING COMPLETED")
    print("Best validation accuracy:", best_accuracy)
    print("Model saved:", MODEL_PATH)
    print("History saved:", HISTORY_PATH)


if __name__ == "__main__":
    main()