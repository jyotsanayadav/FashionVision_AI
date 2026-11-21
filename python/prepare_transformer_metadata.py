import json
import os
import random
from collections import defaultdict

ANNOTATION_FILE = "dataset/modanet-annotations/annotations/modanet2018_instances_train.json"
OUTPUT_FILE = "results/transformer_metadata.json"
SPLIT_FILE = "results/transformer_metadata_split.json"

SEED = 42

random.seed(SEED)


def main():
    print("Loading ModaNet annotations...")

    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    images = {
        image["id"]: image
        for image in data["images"]
    }

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    grouped = defaultdict(list)

    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue

        image_id = annotation["image_id"]

        if image_id not in images:
            continue

        x, y, width, height = annotation["bbox"]

        image = images[image_id]
        image_width = image["width"]
        image_height = image["height"]

        # Normalize bounding-box geometry.
        center_x = (x + width / 2) / image_width
        center_y = (y + height / 2) / image_height
        normalized_width = width / image_width
        normalized_height = height / image_height

        area_fraction = (
            annotation["area"] / (image_width * image_height)
        )

        aspect_ratio = width / max(height, 1)

        grouped[image_id].append({
            "features": [
                center_x,
                center_y,
                normalized_width,
                normalized_height,
                area_fraction,
                aspect_ratio
            ],
            "label": annotation["category_id"]
        })

    # Keep images containing at least one garment.
    image_ids = list(grouped.keys())
    random.shuffle(image_ids)

    total = len(image_ids)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train": image_ids[:train_end],
        "validation": image_ids[train_end:val_end],
        "test": image_ids[val_end:]
    }

    metadata = {
        str(image_id): {
            "file_name": images[image_id]["file_name"],
            "objects": grouped[image_id]
        }
        for image_id in image_ids
    }

    os.makedirs("results", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(metadata, f)

    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f)

    print("\nPREPROCESSING COMPLETE")
    print("----------------------")
    print("Images:", total)
    print("Categories:", len(categories))
    print("Features per garment: 6")

    for name, ids in splits.items():
        print(f"{name.capitalize()} images: {len(ids)}")

    print("\nMetadata saved:", OUTPUT_FILE)
    print("Splits saved:", SPLIT_FILE)


if __name__ == "__main__":
    main()