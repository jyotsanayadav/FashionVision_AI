\# FashionVision AI — Model Evaluation



\## 1. Overview



FashionVision AI is a multi-model fashion recognition and generative AI project. It combines image classification, garment detection, segmentation, metadata classification and synthetic image generation.



This document presents the experimental results, performance benchmarks, observations and limitations of the implemented models.



The project includes:



\- ResNet-18 for fashion image classification.

\- Mask R-CNN for garment detection and segmentation.

\- Transformer for geometric garment metadata classification.

\- Variational Autoencoder (VAE) for image generation and reconstruction.

\- Generative Adversarial Network (GAN) for synthetic fashion image generation.

\- C++ and Python ONNX Runtime inference implementations.



\---



\## 2. Classification Evaluation



\### 2.1 ResNet-18



\*\*Dataset:\*\* Fashion-MNIST  

\*\*Task:\*\* 10-class fashion image classification.



Overall performance:



| Metric | Result |

|---|---:|

| Test Accuracy | 92.39% |

| Macro F1-score | 92.33% |



Per-class performance:



| Class | Precision | Recall | F1-score |

|---|---:|---:|---:|

| T-shirt/top | 84.06% | 91.20% | 87.48% |

| Trouser | 99.39% | 98.50% | 98.95% |

| Pullover | 87.45% | 90.60% | 89.00% |

| Dress | 92.81% | 92.90% | 92.85% |

| Coat | 87.28% | 89.20% | 88.23% |

| Sandal | 98.70% | 98.80% | 98.75% |

| Shirt | 81.85% | 71.70% | 76.44% |

| Sneaker | 94.25% | 98.40% | 96.28% |

| Bag | 99.29% | 98.10% | 98.69% |

| Ankle boot | 98.85% | 94.50% | 96.63% |



\*\*Observations:\*\*



\- ResNet-18 achieved strong classification performance.

\- Trouser, sandal and bag categories achieved particularly high F1-scores.

\- Shirt classification was comparatively difficult, with a recall of 71.70%.

\- Similar visual characteristics between shirts, coats and other upper-body garments may contribute to classification errors.



\### 2.2 Fashion Transformer



\*\*Dataset:\*\* ModaNet garment metadata  

\*\*Task:\*\* Geometric garment metadata classification.



Overall performance:



| Metric | Result |

|---|---:|

| Test Accuracy | 85.26% |

| Macro F1-score | 79.78% |



Per-class F1-scores:



| Category | F1-score |

|---|---:|

| Bag | 85.18% |

| Belt | 93.43% |

| Boots | 68.29% |

| Footwear | 95.67% |

| Outer | 74.39% |

| Dress | 81.20% |

| Sunglasses | 92.43% |

| Pants | 90.89% |

| Top | 79.65% |

| Shorts | 63.60% |

| Skirt | 71.75% |

| Headwear | 89.76% |

| Scarf/tie | 50.88% |



\*\*Observations:\*\*



\- The Transformer achieved 85.26% accuracy.

\- Footwear and belt categories achieved strong F1-scores.

\- Scarf/tie and shorts were comparatively challenging categories.

\- The model uses geometric metadata rather than image pixels.



Therefore, this experiment evaluates metadata-based classification, not direct visual recognition.



\---



\## 3. Garment Detection and Segmentation Evaluation



\### 3.1 Mask R-CNN



Mask R-CNN was implemented for garment detection and instance segmentation using ModaNet annotations.



Two approaches were evaluated:



1\. Training from scratch.

2\. Transfer learning using COCO-pretrained weights.



The pilot experiment used 200 training images and 3 epochs.



\### Overall Results



| Model | Box AP50 | Mask AP50 |

|---|---:|---:|

| Mask R-CNN (Scratch) | 0.0068 | 0.0019 |

| Mask R-CNN (Transfer) | 0.1985 | 0.1372 |



\### Per-category Transfer Learning Results



| Category | Box AP | Mask AP |

|---|---:|---:|

| Bag | 0.3355 | 0.4180 |

| Belt | 0.0268 | 0.0000 |

| Boots | 0.0376 | 0.0702 |

| Footwear | 0.1290 | 0.0829 |

| Outer | 0.6040 | 0.0878 |

| Dress | 0.3239 | 0.2225 |

| Sunglasses | 0.0000 | 0.0000 |

| Pants | 0.5153 | 0.4737 |

| Top | 0.2492 | 0.1315 |

| Shorts | 0.0800 | 0.0800 |

| Skirt | 0.2167 | 0.2175 |

| Headwear | 0.0625 | 0.0000 |

| Scarf/tie | 0.0000 | 0.0000 |



\*\*Observations:\*\*



\- Transfer learning significantly improved performance over training from scratch.

\- Pants, bags and dresses showed comparatively better segmentation performance.

\- Several categories achieved very low AP.

\- The limited training duration and small pilot dataset restrict generalization.



\*\*Conclusion:\*\* The transfer-learning implementation demonstrates garment detection and segmentation functionality, but requires further training and evaluation before production use.



\---



\## 4. Generative Model Evaluation



\### 4.1 Variational Autoencoder (VAE)



The VAE was trained on Fashion-MNIST images.



Its main objectives were:



\- Learning a latent representation of fashion images.

\- Reconstructing input images.

\- Generating new synthetic fashion images.



\*\*Recorded result:\*\*



| Metric | Result |

|---|---:|

| Best validation loss | 238.4394 |

| Loss function | BCE + KL divergence |



Generated images and reconstruction results are available in:



\- `results/vae\_generated\_images.png`

\- `results/vae\_reconstruction.png`



The VAE successfully demonstrates image reconstruction and synthetic image generation.



The validation loss is reported as a training objective and should not be interpreted as classification accuracy.



\### 4.2 Generative Adversarial Network (GAN)



The GAN was trained to generate synthetic Fashion-MNIST images.



Its architecture consists of:



\- Generator: Produces synthetic fashion images.

\- Discriminator: Learns to distinguish real images from generated images.



\*\*Training configuration:\*\*



| Parameter | Value |

|---|---|

| Training epochs | 10 |

| Dataset | Fashion-MNIST |

| Quantitative generation metric | Not calculated |



Generated samples and training visualizations:



\- `results/gan\_generated\_100.png`

\- `results/gan\_loss\_curve.png`

\- `results/gan\_epoch\_01.png` to `results/gan\_epoch\_10.png`



The GAN demonstrates synthetic image generation and training progression.



Quantitative image-generation evaluation remains future work.



\---



\## 5. Python vs C++ Performance Benchmark



The ResNet-18 ONNX model was evaluated using Python and C++ implementations.



\*\*Benchmark configuration:\*\*



| Parameter | Value |

|---|---|

| Model | ResNet-18 Fashion-MNIST |

| Runtime | ONNX Runtime 1.32.0 |

| Execution device | CPU |

| Images tested | 1,000 |

| Batch size | 32 |



\### Benchmark Results



| Metric | Python | C++ |

|---|---:|---:|

| Total inference time (ms) | 2682.12 | 6211.80 |

| Average latency (ms/image) | 2.68 | 6.21 |

| Single-image throughput (images/sec) | 372.84 | 160.98 |

| Batch inference time (ms) | 1530.41 | 2531.28 |

| Batch throughput (images/sec) | 653.42 | 395.06 |



\*\*Observations:\*\*



\- Python achieved lower measured inference latency.

\- Python also achieved higher throughput in both single-image and batch inference.

\- Both implementations successfully performed ONNX-based inference.

\- The results reflect the current implementations and benchmark environment, not an inherent performance advantage of Python over C++.



Further optimization of the C++ implementation may improve its performance.



\---



\## 6. Overall Model Comparison



| Model | Primary Task | Main Result |

|---|---|---|

| ResNet-18 | Image classification | 92.39% accuracy |

| Fashion Transformer | Metadata classification | 85.26% accuracy |

| Mask R-CNN (Scratch) | Detection and segmentation | Box AP50: 0.0068 |

| Mask R-CNN (Transfer) | Detection and segmentation | Box AP50: 0.1985 |

| VAE | Image generation | Validation loss: 238.4394 |

| GAN | Image generation | 10 training epochs |



These models perform different tasks. Their metrics are therefore not directly comparable.



\---



\## 7. Limitations



The current implementation has several limitations:



1\. \*\*Dataset limitations:\*\* The experiments use Fashion-MNIST and ModaNet, which have different image characteristics and tasks.



2\. \*\*Mask R-CNN training:\*\* The detection model was evaluated using a limited pilot training setup of 200 images and 3 epochs.



3\. \*\*Transformer input:\*\* The Transformer uses geometric garment metadata rather than raw image pixels.



4\. \*\*Transformer inference:\*\* Its Streamlit integration derives approximate geometric features from Mask R-CNN predictions. These inputs differ from the ground-truth annotations used during training, so application predictions should be considered experimental.



5\. \*\*Generative evaluation:\*\* Quantitative image-generation metrics such as FID have not yet been calculated for the VAE and GAN.



6\. \*\*Benchmark environment:\*\* Python and C++ performance depends on implementation details, hardware and runtime configuration.



\---



\## 8. Future Improvements



Potential improvements include:



\- Train Mask R-CNN on a larger dataset for more epochs.

\- Improve garment segmentation and category-wise detection performance.

\- Evaluate the Transformer using more consistent inference features and sequence ordering.

\- Calculate FID or other suitable generative-model evaluation metrics.

\- Optimize C++ ONNX inference.

\- Evaluate robustness under different image sizes, lighting conditions and backgrounds.

\- Explore larger and more diverse fashion datasets.



\---



\## 9. Conclusion



FashionVision AI demonstrates the implementation of multiple machine learning and deep learning approaches within one integrated fashion-analysis application.



The project combines classification, object detection, instance segmentation, metadata classification and generative modeling.



ResNet-18 achieved 92.39% classification accuracy, while the Transformer achieved 85.26% metadata classification accuracy. Transfer learning improved Mask R-CNN detection performance compared with training from scratch.



The VAE and GAN demonstrate fashion image reconstruction and synthetic image generation. Python and C++ ONNX implementations also provide an opportunity to study deployment and inference performance.



Overall, the project demonstrates an end-to-end experimental framework for exploring different AI techniques in fashion recognition and virtual cataloging.

