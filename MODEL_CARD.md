\# FashionVision AI — Model Cards



\## 1. Project Overview



FashionVision AI contains five major deep learning models designed for fashion image classification, garment detection, segmentation, metadata classification and image generation.



These models were developed and evaluated using Fashion-MNIST and ModaNet datasets.



The following sections describe each model, its intended use, evaluation results and limitations.



\---



\# 2. ResNet-18 — Fashion Image Classification



\### Model Information

\- Model: ResNet-18

\- Task: Image classification

\- Framework: PyTorch

\- Deployment format: ONNX

\- Runtime: ONNX Runtime

\- Dataset: Fashion-MNIST



\### Architecture



ResNet-18 is a convolutional neural network that uses residual connections to improve information and gradient flow during training.



It classifies an input image into one of ten Fashion-MNIST categories.



\### Training and Evaluation

\- Input image size: 28 × 28 pixels.

\- Input channels: 1 (grayscale).

\- Number of classes: 10.

\- Evaluation dataset: Fashion-MNIST test set.



\### Performance



| Metric | Result |

|---|---:|

| Test Accuracy | 92.39% |

| Macro F1 Score | 92.33% |

| Python ONNX Average Latency | 2.68 ms/image |

| Python ONNX Throughput | 372.84 images/second |



\### Intended Use

\- Fashion-MNIST image classification.

\- Educational demonstrations of CNNs.

\- Testing ONNX model deployment.

\- CPU inference experiments.



\### Limitations

\- Trained on small grayscale Fashion-MNIST images.

\- Not directly trained for high-resolution real-world fashion photographs.

\- Real-world image predictions may not be reliable.

\- Performance can vary with preprocessing and hardware.



\---



\# 3. Mask R-CNN — Garment Detection and Segmentation



\### Model Information

\- Model: Mask R-CNN

\- Task: Object detection and instance segmentation.

\- Framework: PyTorch / Torchvision.

\- Dataset: ModaNet.

\- Training approach: Transfer learning and training from scratch.



\### Architecture



Mask R-CNN is a two-stage object detection architecture.



It identifies individual objects and predicts:

\- Bounding boxes.

\- Object categories.

\- Pixel-level segmentation masks.



The project uses a COCO-pretrained model for transfer learning and compares it with a model trained from scratch.



\### Training and Evaluation

\- Training samples in the limited pilot: 200.

\- Training epochs: 3.

\- Evaluation: Custom 100-image validation set.



\### Performance



| Metric | Scratch | Transfer Learning |

|---|---:|---:|

| Box AP50 | 0.0068 | 0.1985 |

| Mask AP50 | 0.0019 | 0.1372 |



\### Intended Use

\- Experimental garment detection.

\- Garment localization.

\- Instance segmentation.

\- Visual demonstrations of object detection.



\### Limitations

\- Training used a small pilot dataset.

\- Detection and segmentation accuracy remain limited.

\- Performance varies significantly between garment categories.

\- More training data and epochs are required.

\- Not suitable for production-level garment detection in its current state.



\---



\# 4. Fashion Transformer — Metadata Classification



\### Model Information

\- Model: Fashion Transformer.

\- Task: Geometric garment metadata classification.

\- Framework: PyTorch.

\- Dataset: ModaNet-derived metadata.



\### Architecture



The Transformer processes a sequence of geometric garment features.



Each garment is represented using six numerical features:



1\. Center X coordinate.

2\. Center Y coordinate.

3\. Normalized width.

4\. Normalized height.

5\. Area fraction.

6\. Log aspect ratio.



The model learns relationships between these features using Transformer-based attention.



\### Training and Evaluation

\- Training samples: 41,803.

\- Validation samples: 5,225.

\- Test samples: 5,226.

\- Random seed: 42.



\### Performance



| Metric | Result |

|---|---:|

| Test Accuracy | 85.26% |

| Macro F1 Score | 79.78% |



\### Intended Use

\- Experimental garment metadata classification.

\- Learning relationships between garment geometry and categories.

\- Demonstrating Transformer-based sequence modelling.



\### Limitations

\- Uses geometric metadata rather than raw image pixels.

\- Depends on the quality of bounding boxes and segmentation masks.

\- Predictions can be affected by object ordering.

\- In application inference, features are extracted from Mask R-CNN predictions, which may differ from the original annotation features.

\- It should not be considered an independent visual recognition model.



\---



\# 5. Variational Autoencoder (VAE) — Fashion Image Generation



\### Model Information

\- Model: Variational Autoencoder.

\- Task: Image generation and reconstruction.

\- Framework: PyTorch.

\- Dataset: Fashion-MNIST.



\### Architecture



A Variational Autoencoder consists of:

\- An encoder.

\- A latent representation.

\- A reparameterization mechanism.

\- A decoder.



The encoder learns a probabilistic latent representation of an input image. The decoder uses this representation to reconstruct or generate images.



\### Training and Evaluation



The training objective combines:

\- Reconstruction loss.

\- Kullback-Leibler (KL) divergence.



Best recorded validation loss:



\*\*238.4394 (BCE + KL per image).\*\*



\### Intended Use

\- Fashion image generation.

\- Image reconstruction.

\- Latent-space experimentation.

\- Educational demonstrations of generative modelling.



\### Limitations

\- Trained on small grayscale Fashion-MNIST images.

\- Generated images may lack fine visual details.

\- The reported validation loss is not a direct measure of perceptual image quality.

\- More extensive evaluation is needed for generation quality.



\---



\# 6. Generative Adversarial Network (GAN) — Fashion Image Generation



\### Model Information

\- Model: Generative Adversarial Network.

\- Task: Fashion image generation.

\- Framework: PyTorch.

\- Dataset: Fashion-MNIST.



\### Architecture



A GAN consists of two neural networks:



\*\*Generator:\*\* Produces synthetic fashion images.



\*\*Discriminator:\*\* Attempts to distinguish generated images from real training images.



Both networks compete during adversarial training, encouraging the generator to produce increasingly realistic samples.



\### Training and Evaluation

\- Training epochs: 10.

\- Generated sample images: 100.

\- Loss history recorded.

\- Epoch-wise visualizations generated.



\### Intended Use

\- Experimental fashion image generation.

\- Studying adversarial training.

\- Comparing generative approaches.

\- Visual demonstrations of synthetic image creation.



\### Performance Assessment



The model successfully completed its 10-epoch training process and generated fashion image samples.



No quantitative generation-quality metric, such as FID, has been calculated.



\### Limitations

\- Limited training duration.

\- Generated image quality has not been quantitatively validated.

\- Possible instability during adversarial training.

\- Requires additional experimentation and evaluation before practical deployment.



\---



\# 7. Overall Model Comparison



| Model | Primary Task | Main Evaluation Result |

|---|---|---|

| ResNet-18 | Image Classification | 92.39% Accuracy |

| Mask R-CNN | Detection and Segmentation | Box AP50: 0.1985 |

| Fashion Transformer | Metadata Classification | 85.26% Accuracy |

| VAE | Image Generation | Validation loss: 238.4394 |

| GAN | Image Generation | 10 training epochs |



The models address different tasks, so their metrics should not be directly compared against one another.



\---



\# 8. Ethical Considerations and Intended Deployment



FashionVision AI is an academic and experimental project.



Its models are intended for:

\- Educational demonstrations.

\- Research experimentation.

\- Controlled pilot testing.

\- Understanding deep learning applications in fashion technology.



The system has not undergone extensive real-world validation, production security testing or large-scale deployment testing.



It should not be used as a fully validated commercial fashion-recognition service.



Dataset licensing, image redistribution rights and potential dataset biases should be considered before further distribution or commercial use.



\---



\# 9. Future Improvements



1\. Increase Mask R-CNN training data and training duration.

2\. Improve garment detection and segmentation accuracy.

3\. Explore more robust geometric feature extraction for the Transformer.

4\. Evaluate ResNet-18 on a suitable real-world fashion image dataset.

5\. Calculate quantitative generation metrics for VAE and GAN.

6\. Measure memory consumption and end-to-end inference performance.

7\. Test model robustness across different image conditions.



\---



\# 10. Conclusion



FashionVision AI demonstrates five different deep learning approaches within a single fashion-focused application.



The project combines classification, detection, segmentation, metadata learning and generative modelling.



ResNet-18 achieved the strongest classification results, while Mask R-CNN and the Transformer provide experimental garment analysis capabilities. VAE and GAN demonstrate fashion image generation.



Further training, evaluation and real-world validation are necessary before production deployment.

