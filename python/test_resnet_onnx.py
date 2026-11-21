import struct
import numpy as np
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "resnet18_fashion_mnist.onnx"
DATASET_DIR = ROOT / "dataset" / "fashion-mnist" 

IMAGE_FILE = DATASET_DIR / "t10k-images-idx3-ubyte"
LABEL_FILE = DATASET_DIR / "t10k-labels-idx1-ubyte"

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_test_image(index=0):
    with open(IMAGE_FILE, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(count, rows, cols)

    with open(LABEL_FILE, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    image = images[index].astype(np.float32) / 255.0
    label = int(labels[index])

    return image.reshape(1, 1, 28, 28), label


def main():
    print("FASHIONVISION AI - ONNX INFERENCE TEST")
    print("--------------------------------------")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    image, actual_label = load_test_image()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run(
        [output_name],
        {input_name: image}
    )[0]

    probabilities = np.exp(
        predictions[0] - np.max(predictions[0])
    )
    probabilities /= probabilities.sum()

    predicted_label = int(np.argmax(probabilities))

    print("\nModel loaded successfully!")
    print("Input shape:", image.shape)
    print("Output shape:", predictions.shape)

    print("\nACTUAL CLASS:", CLASSES[actual_label])
    print("PREDICTED CLASS:", CLASSES[predicted_label])
    print("CONFIDENCE:", f"{probabilities[predicted_label] * 100:.2f}%")

    print("\nONNX INFERENCE TEST COMPLETED!")


if __name__ == "__main__":
    main()