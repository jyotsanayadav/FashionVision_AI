import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from train_transformer import (
    FashionMetadataDataset,
    FashionTransformer,
    collate_fn,
    NUM_CLASSES,
    BATCH_SIZE
)


# ---------------- CONFIGURATION ----------------

METADATA_FILE = "results/transformer_metadata.json"
SPLIT_FILE = "results/transformer_metadata_split.json"

MODEL_PATH = "models/fashion_transformer.pth"

REPORT_PATH = "results/transformer_classification_report.csv"
CONFUSION_PATH = "results/transformer_confusion_matrix.csv"

CATEGORY_NAMES = [
    "bag", "belt", "boots", "footwear", "outer",
    "dress", "sunglasses", "pants", "top", "shorts",
    "skirt", "headwear", "scarf/tie"
]

# Use CPU for evaluation to avoid XPU memory errors.
DEVICE = torch.device("cpu")


# ---------------- MAIN EVALUATION ----------------

def main():

    print("FASHIONVISION AI - TRANSFORMER EVALUATION")
    print("------------------------------------------")
    print("Device:", DEVICE)

    print("\nLoading metadata...")

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    test_dataset = FashionMetadataDataset(
        metadata,
        splits["test"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print("Test images:", len(test_dataset))

    # Load trained Transformer on CPU.
    model = FashionTransformer().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully!")

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    all_labels = []
    all_predictions = []

    total_loss = 0

    # ---------------- TESTING ----------------

    with torch.no_grad():

        for step, (features, labels, padding_mask) in enumerate(test_loader):

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)

            outputs = model(
                features,
                padding_mask
            )

            loss = criterion(
                outputs.reshape(-1, NUM_CLASSES),
                labels.reshape(-1)
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=-1)

            valid = labels != -100

            all_labels.extend(
                labels[valid].tolist()
            )

            all_predictions.extend(
                predictions[valid].tolist()
            )

            if (step + 1) % 100 == 0:
                print(
                    f"Evaluated batches: "
                    f"{step + 1}/{len(test_loader)}"
                )

    # ---------------- METRICS ----------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    test_loss = total_loss / len(test_loader)

    print("\nTEST RESULTS")
    print("--------------------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # ---------------- CLASSIFICATION REPORT ----------------

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CATEGORY_NAMES,
        output_dict=True,
        zero_division=0
    )

    with open(REPORT_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Category",
            "Precision",
            "Recall",
            "F1 Score",
            "Support"
        ])

        for category in CATEGORY_NAMES:

            row = report[category]

            writer.writerow([
                category,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"]
            ])

    # ---------------- CONFUSION MATRIX ----------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    with open(CONFUSION_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["Actual/Predicted"] + CATEGORY_NAMES
        )

        for i, row in enumerate(matrix):

            writer.writerow(
                [CATEGORY_NAMES[i]] + row.tolist()
            )

    # ---------------- CATEGORY RESULTS ----------------

    print("\nCATEGORY PERFORMANCE")
    print("--------------------------------")

    for category in CATEGORY_NAMES:

        print(
            f"{category:12s} "
            f"F1: {report[category]['f1-score']:.4f}"
        )

    print("\nEVALUATION COMPLETED")

    print("Classification report:", REPORT_PATH)
    print("Confusion matrix:", CONFUSION_PATH)


if __name__ == "__main__":
    main()