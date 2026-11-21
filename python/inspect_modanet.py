import json
from PIL import Image, ImageDraw

annotation_path = "dataset/modanet-annotations/annotations/modanet2018_instances_train.json"
image_dir = "dataset/images"

with open(annotation_path, "r") as f:
    data = json.load(f)

categories = {
    c["id"]: c["name"] for c in data["categories"]
}

annotation = data["annotations"][0]

image_info = next(
    img for img in data["images"]
    if img["id"] == annotation["image_id"]
)

image_path = f"{image_dir}/{image_info['file_name']}"

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Draw bounding box
x, y, w, h = annotation["bbox"]
draw.rectangle(
    [x, y, x + w, y + h],
    outline="red",
    width=4
)

# Draw segmentation polygon
for polygon in annotation["segmentation"]:
    points = list(zip(polygon[::2], polygon[1::2]))
    if len(points) >= 3:
        draw.line(points + [points[0]], fill="lime", width=3)

# Save visualization
output_path = "results/modanet_sample_annotation.jpg"
image.save(output_path)

print("Image:", image_info["file_name"])
print("Category:", categories[annotation["category_id"]])
print("Bounding box:", annotation["bbox"])
print("Saved:", output_path)