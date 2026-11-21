# FashionVision AI (Project 48): C++ core, Tasks 1-5 part 1

## Layout
- include/, src/   : library `fv_core` (tensor, SVD, gradient check, SVG plots, neurons, MLP, dataset, optimizers, trainer)
- tests/           : test_tensor, test_linalg, test_mlp, test_optim
- src/*_demo.cpp   : tensor_demo, mlp_demo, optim_demo
- dataset/fashion-mnist/ : put the 4 unzipped IDX files here
  (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte)

## Build (Windows, MinGW or MSYS2, CMake)
    cmake -S . -B build -G "MinGW Makefiles"
    cmake --build build
    ctest --test-dir build --output-on-failure

## Run (from the project root)
    .\build\tensor_demo.exe
    .\build\mlp_demo.exe
    .\build\optim_demo.exe dataset/fashion-mnist results 4000    # quick run
    .\build\optim_demo.exe                                       # 12000 images, a few minutes

Outputs (SVG charts, CSV, JSON configs) go to results/.
## ONNX Runtime Integration and Performance Benchmark

FashionVision AI supports model deployment using ONNX Runtime with both Python and C++.

### ResNet-18 ONNX Model

- Architecture: ResNet-18
- Dataset: Fashion-MNIST
- Input shape: 1 × 28 × 28
- Output classes: 10
- Test accuracy: 92.39%
- Macro F1-score: 92.33%

### C++ ONNX Inference

Implemented C++ inference using ONNX Runtime.

Features:
- ONNX model loading
- Single-image inference
- Full 10,000-image test evaluation
- Confusion matrix generation
- Classification report generation

### Python vs C++ Performance Benchmark

Both implementations were benchmarked using 1,000 Fashion-MNIST test images on CPU.

| Metric | Python | C++ |
|---|---:|---:|
| Average latency (ms/image) | 2.68 | 6.21 |
| Throughput (images/sec) | 372.84 | 160.98 |
| Batch size | 32 | 32 |
| Batch throughput (images/sec) | 653.42 | 395.06 |

**Benchmark observations:**
- Batch inference achieved higher throughput than single-image inference.
- Python showed better measured throughput in this benchmark.
- Both implementations use ONNX Runtime 1.32.0.

Note: These are initial measurements. Runtime threading and benchmarking conditions can affect performance.

### Benchmark Results

Results are available in:

- `results/onnx_benchmark_comparison.csv`
- `results/cpp_confusion_matrix.csv`
- `results/cpp_classification_report.csv`
- `results/resnet18_confusion_matrix.csv`
- `results/resnet18_classification_report.csv`

### Running C++ ONNX Inference

Build using the MSVC CMake build:

```powershell
cmake --build ..\build_msvc --target resnet_onnx
```

Run:

```powershell
..\build_msvc\resnet_onnx.exe
```

### Running C++ ONNX Benchmark

Build:

```powershell
cmake --build ..\build_msvc --target benchmark_onnx
```

Run:

```powershell
..\build_msvc\benchmark_onnx.exe
```
---

## FashionVision AI — Deep Learning Application

FashionVision AI is an integrated fashion recognition and virtual cataloging system combining computer vision, deep learning, geometric metadata classification and generative AI.

### Implemented AI Models

| Model | Function | Dataset |
|---|---|---|
| ResNet-18 | Fashion image classification | Fashion-MNIST |
| Mask R-CNN | Garment detection and instance segmentation | ModaNet |
| Fashion Transformer | Geometric garment metadata classification | ModaNet |
| VAE | Fashion image reconstruction and generation | Fashion-MNIST |
| GAN | Synthetic fashion image generation | Fashion-MNIST |

### Streamlit Application

The project includes an interactive Streamlit interface with four sections:

1. **Garment Detection:** Upload fashion images and visualize garment detection, segmentation and experimental Transformer metadata predictions.
2. **Fashion Classification:** Classify fashion images using the trained ResNet-18 ONNX model.
3. **Generative Models:** Explore VAE reconstructions, generated images and GAN training outputs.
4. **Model Information:** View information about the implemented deep learning models.

### Running the Application

Activate the Python virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start Streamlit:

```powershell
streamlit run app.py
```

Open the local application at:

```text
http://localhost:8501
```

### Evaluation and Results

Detailed experimental results, model comparisons, benchmark measurements and limitations are documented in:

**[EVALUATION.md](EVALUATION.md)**

Important results:

| Model | Evaluation Result |
|---|---:|
| ResNet-18 | 92.39% test accuracy |
| Fashion Transformer | 85.26% test accuracy |
| Mask R-CNN (Transfer) | 0.1985 Box AP50 |
| Mask R-CNN (Transfer) | 0.1372 Mask AP50 |

Python and C++ ONNX inference implementations are also benchmarked in the project.

### Project Structure

```text
FashionVision_AI/
│
├── app.py
├── CMakeLists.txt
├── README.md
├── EVALUATION.md
│
├── include/       # C++ header files
├── src/           # C++ implementations and demos
├── python/        # Python training and evaluation scripts
├── models/        # Saved model checkpoints
├── dataset/       # Dataset files
├── results/       # Evaluation reports and visualizations
├── tests/         # C++ tests
├── assets/        # Project assets
├── docs/          # Additional documentation
└── third_party/   # External dependencies
```

### Current Limitations

- Mask R-CNN was evaluated using a limited pilot training setup and requires further training.
- Transformer predictions use geometric metadata rather than direct visual features.
- VAE and GAN quantitative generation metrics remain future work.
- Python and C++ benchmark results reflect the current implementations and hardware environment.

---