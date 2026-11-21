import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

benchmark = [
    {
        "Model": "ResNet-18",
        "Task": "Fashion-MNIST Image Classification",
        "Accuracy": "92.39%",
        "Macro F1": "92.33%",
        "Box AP50": "N/A",
        "Mask AP50": "N/A",
        "Notes": "10-class image classification"
    },
    {
        "Model": "Transformer",
        "Task": "Garment Metadata Classification",
        "Accuracy": "85.26%",
        "Macro F1": "79.78%",
        "Box AP50": "N/A",
        "Mask AP50": "N/A",
        "Notes": "Uses geometric metadata, not image pixels"
    },
    {
        "Model": "Mask R-CNN (Scratch)",
        "Task": "Garment Detection and Segmentation",
        "Accuracy": "N/A",
        "Macro F1": "N/A",
        "Box AP50": "0.0068",
        "Mask AP50": "0.0019",
        "Notes": "200 training images, 3 epochs"
    },
    {
        "Model": "Mask R-CNN (Transfer)",
        "Task": "Garment Detection and Segmentation",
        "Accuracy": "N/A",
        "Macro F1": "N/A",
        "Box AP50": "0.1985",
        "Mask AP50": "0.1372",
        "Notes": "COCO pretrained; 200 training images, 3 epochs"
    },
    {
        "Model": "VAE",
        "Task": "Fashion Image Generation",
        "Accuracy": "N/A",
        "Macro F1": "N/A",
        "Box AP50": "N/A",
        "Mask AP50": "N/A",
        "Notes": "Best validation loss: 238.4394 (BCE + KL per image)"
    },
    {
        "Model": "GAN",
        "Task": "Fashion Image Generation",
        "Accuracy": "N/A",
        "Macro F1": "N/A",
        "Box AP50": "N/A",
        "Mask AP50": "N/A",
        "Notes": "10 training epochs; quantitative generation metric not yet calculated"
    }
]

output = RESULTS / "model_benchmark_summary.csv"

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=benchmark[0].keys())
    writer.writeheader()
    writer.writerows(benchmark)

print("FASHIONVISION AI - MODEL BENCHMARK SUMMARY")
print("------------------------------------------")

for item in benchmark:
    print(f"\nModel: {item['Model']}")
    print(f"Task: {item['Task']}")
    print(f"Accuracy: {item['Accuracy']}")
    print(f"Macro F1: {item['Macro F1']}")
    print(f"Box AP@0.5: {item['Box AP50']}")
    print(f"Mask AP@0.5: {item['Mask AP50']}")

print("\nBenchmark summary saved:", output)
print("BENCHMARK SUMMARY COMPLETED!")