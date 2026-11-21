import json
from collections import Counter, defaultdict

ANNOTATION_FILE = (
    "dataset/modanet-annotations/annotations/"
    "modanet2018_instances_train.json"
)

SPLIT_FILE = "results/modanet_split.json"
TRAIN_SAMPLES = 200


# Load training split
with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

train_ids = set(split["train_ids"][:TRAIN_SAMPLES])

# Load annotations
with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

# Count instances and unique images per category
instance_counts = Counter()
image_counts = defaultdict(set)

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in train_ids:
        continue

    if annotation.get("iscrowd", 0):
        continue

    category_id = annotation["category_id"]

    instance_counts[category_id] += 1
    image_counts[category_id].add(image_id)


# Display results
print("\nMODANET TRAINING DATA DISTRIBUTION")
print("----------------------------------")

print(f"Training images: {len(train_ids)}")
print(f"Total clothing instances: {sum(instance_counts.values())}")

print("\nCATEGORY DISTRIBUTION")
print("-----------------------------------------------")
print(f"{'Category':<15}{'Instances':>12}{'Images':>12}")

for category_id, name in categories.items():

    print(
        f"{name:<15}"
        f"{instance_counts[category_id]:>12}"
        f"{len(image_counts[category_id]):>12}"
    )

print("\nDistribution analysis completed!")