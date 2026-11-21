import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class ModaNetDataset(Dataset):

    def __init__(self, annotation_file, image_dir, image_ids=None):
        self.image_dir = image_dir

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]

        if image_ids is not None:
            self.images = [
                img for img in self.images
                if img["id"] in image_ids
            ]

        self.categories = {
            category["id"]: category["name"]
            for category in data["categories"]
        }

        self.annotations = {}

        for annotation in data["annotations"]:
            if annotation.get("iscrowd", 0):
                continue

            image_id = annotation["image_id"]

            if image_id not in self.annotations:
                self.annotations[image_id] = []

            self.annotations[image_id].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_info = self.images[index]

        image_id = image_info["id"]
        filename = image_info["file_name"]

        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        boxes = []
        labels = []
        masks = []
        areas = []

        annotations = self.annotations.get(image_id, [])

        for annotation in annotations:

            x, y, w, h = annotation["bbox"]

            if w <= 0 or h <= 0:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            mask_image = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask_image)

            for polygon in annotation["segmentation"]:

                points = [
                    (polygon[i], polygon[i + 1])
                    for i in range(0, len(polygon) - 1, 2)
                ]

                if len(points) >= 3:
                    draw.polygon(points, fill=1)

            mask = torch.from_numpy(
                np.array(mask_image)
            )

            if mask.sum() == 0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(annotation["category_id"])
            masks.append(mask)
            areas.append(float(annotation.get("area", w * h)))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks).to(torch.uint8)
            areas = torch.tensor(areas, dtype=torch.float32)

        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": torch.zeros(
                (len(boxes),),
                dtype=torch.int64
            )
        }

        return to_tensor(image), target