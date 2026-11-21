import torch
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "resnet18_fashion_mnist.pth"
ONNX_PATH = ROOT / "models" / "resnet18_fashion_mnist.onnx"


def create_model():
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)

    return model


def main():
    print("FASHIONVISION AI - RESNET ONNX EXPORT")
    print("--------------------------------------")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    print("ResNet-18 checkpoint loaded successfully!")

    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    print("ONNX export successful!")
    print("Saved model:", ONNX_PATH)
    print("Input shape: [batch, 1, 28, 28]")
    print("Output shape: [batch, 10]")
    print("RESNET ONNX EXPORT COMPLETED!")


if __name__ == "__main__":
    main()