\# FashionVision AI — Requirements and Validation



\## 1. Project Objective



FashionVision AI aims to provide an AI-based fashion analysis system that supports fashion image classification, garment detection, segmentation, metadata classification and image generation through an interactive application.



The following requirements define measurable acceptance criteria and compare them against the results obtained during development.



\## 2. Functional and Performance Requirements



| ID | Use Case | Measurable Requirement | Actual Result | Status |

|---|---|---|---|---|

| R1 | Fashion Classification | Test accuracy ≥ 90% and macro F1 ≥ 0.90 | Accuracy: 92.39%, Macro F1: 92.33% | PASS |

| R2 | Garment Detection | Box AP50 ≥ 0.50 | Box AP50: 0.1985 | FAIL |

| R3 | Garment Segmentation | Mask AP50 ≥ 0.40 | Mask AP50: 0.1372 | FAIL |

| R4 | Metadata Classification | Test accuracy ≥ 90% | Accuracy: 85.26% | FAIL |

| R5 | Image Inference | Average inference latency < 10 ms/image | Python ONNX: 2.68 ms/image | PASS\* |

| R6 | Image Generation | Generate and display fashion samples using VAE and GAN | Both models trained and sample images generated | PASS\*\* |

| R7 | Application Interface | Provide working pages for detection, classification, generation and model information | All four pages implemented and tested | PASS |

| R8 | Model Evaluation | Provide accuracy, F1, AP and benchmark results wherever applicable | Evaluation reports and benchmark results generated | PASS |



\*R5 measures ResNet-18 ONNX inference on CPU using the recorded 1,000-image benchmark. It does not represent complete application response time.



\*\*R6 confirms successful training and image generation, not production-level image quality.



\## 3. Experimental Conditions



\- ResNet-18 was evaluated on the Fashion-MNIST test dataset.

\- Mask R-CNN transfer learning was evaluated using a limited pilot of 200 training samples and 3 epochs.

\- Fashion Transformer was evaluated on the ModaNet-derived metadata test split.

\- VAE and GAN were trained using Fashion-MNIST images.

\- ONNX benchmarking was conducted using CPU inference.

\- Performance results apply to the recorded experimental setup and may differ across hardware and datasets.



\## 4. Overall Acceptance Assessment



FashionVision AI successfully demonstrates its major functional components, including classification, detection, segmentation, metadata prediction, image generation and application integration.



However, not all proposed performance targets were achieved. ResNet-18 classification and measured Python ONNX inference latency met their targets. Mask R-CNN and Transformer performance require further improvement.



The system is therefore suitable for an academic demonstration and controlled pilot, but its current experimental results do not justify production deployment.



\## 5. Future Improvements



1\. Train Mask R-CNN on a larger and more representative dataset.

2\. Improve garment detection and segmentation performance.

3\. Improve Transformer feature extraction and inference reliability.

4\. Evaluate generative models using quantitative image-quality metrics.

5\. Measure complete application latency and peak memory consumption.

6\. Conduct additional testing across different hardware and real-world fashion images.

