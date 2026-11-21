import json
import random
import os

ANNOTATION_FILE = (
    "dataset/modanet-annotations/annotations/"
    "modanet2018_instances_train.json"
)

OUTPUT_FILE = "results/modanet_split.json"

SEED = 42
TRAIN_RATIO = 0.90

with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

image_ids = [image["id"] for image in data["images"]]

random.seed(SEED)
random.shuffle(image_ids)

split_index = int(len(image_ids) * TRAIN_RATIO)

train_ids = image_ids[:split_index]
val_ids = image_ids[split_index:]

split = {
    "seed": SEED,
    "train_ids": train_ids,
    "val_ids": val_ids
}

os.makedirs("results", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(split, f)

print("MODANET DATASET SPLIT")
print("Total images:", len(image_ids))
print("Training images:", len(train_ids))
print("Validation images:", len(val_ids))
print("Random seed:", SEED)
print("Saved:", OUTPUT_FILE)

assert len(set(train_ids).intersection(val_ids)) == 0
print("Overlap check: PASSED")