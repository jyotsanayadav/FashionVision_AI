import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

def _image_bytes(image):
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "modanet_maskrcnn_transfer.pth"

ANNOTATION_FILE = (
    ROOT / "dataset" / "modanet-annotations"
    / "annotations" / "modanet2018_instances_train.json"
)

DEVICE = torch.device("cpu")

NUM_CLASSES = 14
CONFIDENCE_THRESHOLD = 0.01
DISPLAY_THRESHOLD = 0.25
MAX_DISPLAY_DETECTIONS = 10
# =====================================================
# FASHION TRANSFORMER
# =====================================================

import math

TRANSFORMER_PATH = ROOT / "models" / "fashion_transformer.pth"

NUM_FEATURES = 6
TRANSFORMER_CLASSES = 13
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_OBJECTS = 30
DROPOUT = 0.1


class FashionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = torch.nn.Linear(NUM_FEATURES, D_MODEL)

        self.position_embedding = torch.nn.Parameter(
            torch.randn(1, MAX_OBJECTS, D_MODEL) * 0.02
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.classifier = torch.nn.Linear(
            D_MODEL,
            TRANSFORMER_CLASSES
        )

    def forward(self, x, padding_mask):

        x = self.embedding(x)

        x = x + self.position_embedding[:, :x.size(1), :]

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        return self.classifier(x)

@st.cache_resource
def load_transformer():

    model = FashionTransformer()

    checkpoint = torch.load(
        TRANSFORMER_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_transformer(boxes, masks, image_size, categories):

    if len(boxes) == 0:
        return []

    image_width, image_height = image_size

    features = []

    for box, mask in zip(boxes[:MAX_OBJECTS], masks[:MAX_OBJECTS]):

        x1, y1, x2, y2 = box

        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)

        center_x = (float(x1) + float(x2)) / 2 / image_width
        center_y = (float(y1) + float(y2)) / 2 / image_height

        normalized_width = width / image_width
        normalized_height = height / image_height

        # Approximate object area using the predicted segmentation mask.
        area_fraction = float(
            np.count_nonzero(mask[0] >= 0.5)
        ) / (image_width * image_height)

        aspect_ratio = width / max(height, 1.0)

        features.append([
            center_x,
            center_y,
            normalized_width,
            normalized_height,
            area_fraction,
            math.log(max(aspect_ratio, 1e-6))
        ])

    model = load_transformer()

    input_tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0)

    padding_mask = torch.zeros(
        1,
        len(features),
        dtype=torch.bool
    )

    with torch.inference_mode():
        outputs = model(input_tensor, padding_mask)
        probabilities = torch.softmax(outputs, dim=-1)

    results = []

    for i in range(len(features)):

        predicted_index = int(
            probabilities[0, i].argmax()
        )

        category_id = predicted_index + 1

        results.append({
            "category": categories.get(
                category_id,
                f"Class {category_id}"
            ),
            "confidence": float(
                probabilities[0, i, predicted_index]
            )
        })

    return results

    model = FashionTransformer()

    checkpoint = torch.load(
        TRANSFORMER_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
# =====================================================
# RESNET-18 CLASSIFICATION
# =====================================================

RESNET_MODEL_PATH = ROOT / "models" / "resnet18_fashion_mnist.onnx"

RESNET_CLASSES = [
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


@st.cache_resource
def load_resnet():

    if not RESNET_MODEL_PATH.exists():
        raise FileNotFoundError("ResNet ONNX model not found.")

    return ort.InferenceSession(
        str(RESNET_MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )


def classify_fashion_image(image):

    session = load_resnet()

    image = image.convert("L").resize((28, 28))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = image_array.reshape(1, 1, 28, 28)

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: image_array}
    )

    logits = outputs[0][0]

    probabilities = np.exp(
        logits - np.max(logits)
    )

    probabilities /= probabilities.sum()

    predicted_index = int(np.argmax(probabilities))

    return (
        RESNET_CLASSES[predicted_index],
        probabilities
    )


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FashionVision AI",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #202B3C;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

        .stDeployButton {
        display: none;
    }

    /* Header */
    .hero {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(120deg, #EEF5FF, #F8FAFF);
        border: 1px solid #DCE7F5;
        margin-bottom: 28px;
    }

    .hero h1 {
        color: #233E63;
        font-size: 38px;
    }

    .hero p {
        color: #526781;
        font-size: 16px;
    }

    /* Information cards */
    .info-card {
        padding: 20px;
        border-radius: 14px;
        background-color: #F8FAFD;
        border: 1px solid #E3EAF3;
        margin-bottom: 14px;
        color: #26364A;
    }

    .info-card h3 {
        color: #315D91;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F5F8FC;
    }
</style>
    """, unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND CATEGORY NAMES
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"ModaNet annotation file not found: {ANNOTATION_FILE}"
        )

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES
    )

    # Same inference configuration as the trained model.
    model.transform.min_size = (320,)
    model.transform.max_size = 320

    model.rpn._pre_nms_top_n = {
        "training": 500,
        "testing": 250
    }

    model.rpn._post_nms_top_n = {
        "training": 100,
        "testing": 50
    }

    model.roi_heads.detections_per_img = 50

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model, categories


# =====================================================
# PREDICTION
# =====================================================

def predict_image(model, image):

    image_tensor = to_tensor(image).to(DEVICE)

    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()

    return boxes, labels, scores, masks


# =====================================================
# VISUALIZATION
# =====================================================

def create_visualization(
    image,
    boxes,
    labels,
    scores,
    masks,
    categories,
    show_masks=True
):

    result = image.convert("RGBA")
    overlay = np.array(result).copy()

    palette = [
        (255, 80, 80),
        (70, 150, 255),
        (70, 200, 120),
        (255, 180, 60),
        (180, 100, 255),
        (255, 100, 180),
        (60, 200, 200),
        (220, 220, 60),
        (120, 150, 255),
        (255, 130, 80),
        (100, 210, 150),
        (200, 120, 220),
        (150, 150, 150)
    ]

    if show_masks:

        for index, label in enumerate(labels):

            mask = masks[index, 0] >= 0.5

            color = palette[
                (int(label) - 1) % len(palette)
            ]

            overlay[mask, :3] = (
                0.55 * overlay[mask, :3]
                + 0.45 * np.array(color)
            ).astype(np.uint8)

    result = Image.fromarray(overlay).convert("RGB")

    draw = ImageDraw.Draw(result)

    for box, label, score in zip(boxes, labels, scores):

        x1, y1, x2, y2 = box

        category_name = categories.get(
            int(label),
            f"Class {label}"
        )

        color = palette[
            (int(label) - 1) % len(palette)
        ]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3
        )

        text = f"{category_name}: {score * 100:.1f}%"

        text_box = draw.textbbox((x1, max(0, y1 - 22)), text)

        draw.rectangle(
            text_box,
            fill=color
        )

        draw.text(
            (x1, max(0, y1 - 22)),
            text,
            fill="white"
        )

    return result


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1> FashionVision AI</h1>
    <p>
        AI-powered fashion recognition, garment detection
        and intelligent visual analysis.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("FashionVision AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Garment Detection",
        "Fashion Classification",
        "Generative Models",
        "Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Project 48 | Deep Learning and Computer Vision"
)


# =====================================================
# GARMENT DETECTION PAGE
# =====================================================

if page == "Garment Detection":

    st.header("Garment Detection & Segmentation")

    st.write(
        "Upload a fashion photograph to identify clothing "
        "items and visualize their segmentation masks."
    )

    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png"]
    )

    show_masks = st.checkbox(
        "Show segmentation masks",
        value=True
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                image,
                use_container_width=True
            )

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Loading model and analyzing image..."
                ):

                    model, categories = load_model()

                    boxes, labels, scores, masks = predict_image(
                        model,
                        image
                    )
                    
                    # Keep model inference threshold unchanged.
                    # Apply a separate filter for cleaner visualization.

                    valid = scores >= DISPLAY_THRESHOLD

                    boxes = boxes[valid]
                    labels = labels[valid]
                    scores = scores[valid]
                    masks = masks[valid]

                    # Display highest-confidence detections first.
                    order = np.argsort(scores)[::-1][:MAX_DISPLAY_DETECTIONS]

                    boxes = boxes[order]
                    labels = labels[order]
                    scores = scores[order]
                    masks = masks[order]




                    result_image = create_visualization(
                        image,
                        boxes,
                        labels,
                        scores,
                        masks,
                        categories,
                        show_masks
                    )
                    transformer_results = predict_transformer(
                        boxes,
                        masks,
                        image.size,
                        categories
                    )

                st.success("Image analysis completed!")
                if transformer_results:
                    st.markdown("---")
                    st.subheader("Transformer Metadata Predictions")
                    st.caption(
                        "Experimental predictions based on geometric features "
                        "from Mask R-CNN boxes and masks, not direct image recognition."
                    )

                    for i, item in enumerate(transformer_results):
                        st.write(
                            f"{i + 1}. {item['category']} — "
                            f"{item['confidence'] * 100:.2f}%"
                        )

                with col2:
                    st.subheader("Detection Results")
                    st.image(
                        result_image,
                        use_container_width=True
                    )

                st.markdown("---")

                st.subheader("Analysis Summary")

                metric1, metric2 = st.columns(2)

                with metric1:
                    st.metric(
                        "Detected Garments",
                        len(scores)
                    )

                with metric2:
                    average_confidence = (
                        float(np.mean(scores)) * 100
                        if len(scores) > 0
                        else 0
                    )

                    st.metric(
                        "Average Confidence",
                        f"{average_confidence:.2f}%"
                    )

                if len(scores) > 0:

                    st.subheader("Detected Clothing Items")

                    for i, (label, score, box) in enumerate(
                        zip(labels, scores, boxes)
                    ):

                        category_name = categories.get(
                            int(label),
                            f"Class {label}"
                        )

                        with st.expander(
                            f"{i + 1}. {category_name} "
                            f"({score * 100:.2f}%)"
                        ):

                            st.write(
                                f"**Category:** {category_name}"
                            )

                            st.write(
                                f"**Confidence:** "
                                f"{score * 100:.2f}%"
                            )

                            st.write(
                                f"**Bounding Box:** "
                                f"{[round(float(v), 2) for v in box]}"
                            )

                    st.download_button(
                        "Download Detection Result",
                        data=__import__("io").BytesIO(
                            _image_bytes(result_image)
                        ),
                        file_name="fashionvision_result.png",
                        mime="image/png"
                    )

                else:
                    st.warning(
                        "No garments were detected in this image."
                    )

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )

    else:

        st.info(
            "Upload a JPG or PNG image to get started."
        )


# =====================================================
# MODEL INFORMATION PAGE
# =====================================================
# =====================================================
# FASHION CLASSIFICATION PAGE
# =====================================================

elif page == "Fashion Classification":

    st.header("Fashion Image Classification")

    st.write(
        "Classify clothing images into 10 Fashion-MNIST categories "
        "using our trained ResNet-18 model."
    )

    st.warning(
        "This model was trained on 28×28 grayscale Fashion-MNIST "
        "images. Uploaded photographs are resized to this format "
        "and may produce unreliable predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload an image for classification",
        type=["jpg", "jpeg", "png"],
        key="resnet_upload"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        if st.button("Classify Image", type="primary"):

            try:

                with st.spinner("Running ResNet-18 inference..."):

                    predicted_class, probabilities = (
                        classify_fashion_image(image)
                    )

                st.success("Classification completed!")

                st.metric(
                    "Predicted Category",
                    predicted_class
                )

                st.metric(
                    "Confidence",
                    f"{probabilities.max() * 100:.2f}%"
                )

                st.subheader("Class Probabilities")

                probability_data = {
                    "Category": RESNET_CLASSES,
                    "Probability": probabilities * 100
                }

                st.bar_chart(
                    probability_data,
                    x="Category",
                    y="Probability"
                )

            except Exception as error:

                st.error(f"Classification failed: {error}")
elif page == "Generative Models":

    st.header("Fashion Image Generation")
    st.write(
        "Explore synthetic fashion images generated using "
        "Variational Autoencoders and Generative Adversarial Networks."
    )

    results_dir = ROOT / "results"

    # ================= VAE =================

    st.markdown("---")
    st.subheader("1. Variational Autoencoder (VAE)")

    st.write(
        "The VAE learns fashion image patterns and generates "
        "new images while reconstructing existing samples."
    )

    vae_col1, vae_col2 = st.columns(2)

    with vae_col1:
        st.caption("Generated Fashion Images")
        st.image(
            str(results_dir / "vae_generated_images.png"),
            use_container_width=True
        )

    with vae_col2:
        st.caption("Fashion Image Reconstruction")
        st.image(
            str(results_dir / "vae_reconstruction.png"),
            use_container_width=True
        )

    # ================= GAN =================

    st.markdown("---")
    st.subheader("2. Generative Adversarial Network (GAN)")

    st.write(
        "The GAN uses a Generator and Discriminator to "
        "learn and produce synthetic fashion images."
    )

    st.caption("100 Generated Fashion Images")

    st.image(
        str(results_dir / "gan_generated_100.png"),
        use_container_width=True
    )

    st.caption("GAN Training Loss")

    st.image(
        str(results_dir / "gan_loss_curve.png"),
        use_container_width=True
    )

    with st.expander("View GAN Training Across 10 Epochs"):

        for epoch in range(1, 11):

            image_path = (
                results_dir / f"gan_epoch_{epoch:02d}.png"
            )

            st.image(
                str(image_path),
                caption=f"Epoch {epoch}",
                use_container_width=True
            )

elif page == "Model Information":

    st.header("Our Deep Learning Models")

    st.write(
        "FashionVision AI combines multiple deep learning "
        "approaches for fashion analysis and experimentation."
    )

    models = [
        (
            "ResNet-18",
            "Fashion-MNIST classification",
            "92.39%",
            "Classifies 10 Fashion-MNIST clothing categories."
        ),
        (
            "Mask R-CNN",
            "Garment detection and segmentation",
            "Transfer-learning pilot",
            "Detects garment regions and generates segmentation masks."
        ),
        (
            "Fashion Transformer",
            "Geometric metadata classification",
            "85.26%",
            "Uses six geometric features to classify garment instances."
        ),
        (
            "Variational Autoencoder",
            "Fashion image generation",
            "Trained",
            "Learns a latent representation of Fashion-MNIST images."
        ),
        (
            "Generative Adversarial Network",
            "Fashion image generation",
            "Trained",
            "Generates experimental Fashion-MNIST-style images."
        )
    ]

    for name, task, performance, description in models:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>{name}</h3>
                <b>{task}</b><br><br>
                <b>Result:</b> {performance}<br>
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "These models solve different tasks and use different "
        "input formats. Their performance figures are not "
        "directly comparable. Mask R-CNN is an experimental "
        "transfer-learning pilot."
    )

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write(
        """
        - Python
        - PyTorch
        - Torchvision
        - ONNX Runtime
        - C++
        - Streamlit
        - NumPy
        """
    )


# =====================================================
# IMAGE EXPORT HELPER
# =====================================================

def _image_bytes(image):

    import io

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()