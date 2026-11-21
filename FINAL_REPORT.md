\# FASHIONVISION AI

\## Fashion Attribute Recognition and Virtual Cataloging Tool



\### 1. Abstract



FashionVision AI is a machine-learning-based application designed to explore fashion image classification, garment detection, segmentation, metadata prediction, and image generation.



The project combines deep learning models, computer vision techniques, ONNX-based inference, and a Streamlit interface. It uses Fashion-MNIST for image classification and generative experiments, and ModaNet for garment detection, segmentation, and metadata experiments.



The ResNet-18 classifier achieved 92.39% test accuracy and 92.33% macro F1-score on Fashion-MNIST. The Mask R-CNN transfer-learning experiment achieved a Box AP50 of 0.1985 and Mask AP50 of 0.1372. A Transformer-based metadata model achieved 85.26% accuracy.



The application also demonstrates Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) for fashion image generation.



The project provides an integrated academic prototype while highlighting the limitations and improvement opportunities of individual models.



\---



\### 2. Introduction



The fashion industry increasingly uses artificial intelligence for product categorization, visual search, inventory organization, and digital catalog management.



Manually organizing large collections of fashion images can be time-consuming. Computer vision provides methods for automatically identifying clothing categories, locating garments, and extracting useful visual information.



FashionVision AI explores these capabilities through multiple machine-learning and deep-learning approaches.



\#### 2.1 Problem Statement



The project addresses the challenge of analyzing and organizing fashion images through automated techniques.



The main problems considered are:



\- Identifying fashion categories from images.

\- Detecting and segmenting garments.

\- Predicting garment metadata.

\- Generating synthetic fashion images.

\- Presenting model outputs through a user-friendly interface.



\#### 2.2 Objectives



1\. Develop an image classification model for fashion categories.

2\. Experiment with object detection and segmentation.

3\. Explore Transformer-based garment metadata prediction.

4\. Implement VAE and GAN image generation.

5\. Integrate trained models into a single application.

6\. Evaluate models using appropriate performance metrics.

7\. Explore efficient CPU inference using ONNX Runtime.



\---



\### 3. Technology Stack



| Category | Technologies |

|---|---|

| Programming languages | Python, C++ |

| Deep learning | PyTorch |

| Computer vision | Torchvision, NumPy |

| Image classification | ResNet-18 |

| Object detection | Mask R-CNN |

| Metadata prediction | Transformer |

| Generative models | VAE, GAN |

| Model deployment | ONNX Runtime |

| Application interface | Streamlit |

| Build system | CMake |

| Development environment | Visual Studio, VS Code |

| Data visualization | Matplotlib |



\---



\### 4. Dataset Description



Two primary datasets were used.



\#### 4.1 Fashion-MNIST



Fashion-MNIST contains grayscale images of ten clothing categories.



\- Training images: 60,000

\- Test images: 10,000

\- Image dimensions: 28 × 28 pixels

\- Channels: 1 (grayscale)

\- Classes: 10



The dataset was used for:

\- ResNet-18 classification.

\- VAE training.

\- GAN training.



Images were converted into floating-point values and normalized to the range \[0, 1].



\#### 4.2 ModaNet



ModaNet is a fashion dataset containing garment annotations.



\- Images represented in the annotation collection: 52,377

\- Annotations: 265,705

\- Categories: 13



It was used for:

\- Mask R-CNN garment detection and segmentation.

\- Transformer-based geometric metadata prediction.



The Mask R-CNN experiment used 47,139 training and 5,238 validation samples.



The Transformer experiment used:

\- Training: 41,803 samples

\- Validation: 5,225 samples

\- Testing: 5,226 samples



A random seed of 42 was used for the documented Transformer split.



\*\*Dataset note:\*\* ModaNet annotations and image files were obtained from different sources. The annotation license does not automatically establish the licensing status of every image. Refer to `DATASET\_CARD.md` for details.



\---



\### 5. System Architecture



FashionVision AI is organized into separate model components connected through a Streamlit application.



\#### 5.1 Main Components



\*\*1. Fashion Classification\*\*



A ResNet-18 model classifies Fashion-MNIST images into ten clothing categories. The trained model is exported to ONNX format for inference.



\*\*2. Garment Detection and Segmentation\*\*



Mask R-CNN identifies garment regions and predicts segmentation masks. Transfer learning with COCO-pretrained weights was explored.



\*\*3. Transformer Metadata Prediction\*\*



A Transformer model predicts garment categories using geometric features extracted from bounding boxes and masks.



The six input features are:

\- Normalized center X.

\- Normalized center Y.

\- Normalized width.

\- Normalized height.

\- Area fraction.

\- Logarithmic aspect ratio.



This is a geometry-based experiment, not direct image-based recognition.



\*\*4. Generative Models\*\*



Two generative approaches were implemented:



\- VAE: Learns a latent representation and generates reconstructed or new fashion images.

\- GAN: Uses a generator and discriminator to learn synthetic fashion image generation.



\*\*5. Application Interface\*\*



Streamlit provides four main sections:



\- Garment Detection.

\- Fashion Classification.

\- Generative Models.

\- Model Information.



Users can upload images, view predictions, and explore generated outputs.



\---



\### 6. Methodology and Implementation



\#### 6.1 Image Classification Using ResNet-18



ResNet-18 was trained using Fashion-MNIST.



Residual connections help information flow through deeper neural networks and support effective training.



The model was evaluated using:

\- Accuracy.

\- Precision.

\- Recall.

\- Macro F1-score.

\- Confusion matrix.



The trained model was exported to ONNX and tested using ONNX Runtime.



\#### 6.2 Garment Detection Using Mask R-CNN



Mask R-CNN was used to identify garment objects and generate pixel-level segmentation masks.



Two experiments were conducted:



1\. Training from scratch.

2\. Transfer learning using pretrained COCO weights.



The experiments used a limited pilot training setup of 200 samples and three epochs.



Evaluation metrics included:

\- Bounding Box AP50.

\- Mask AP50.

\- Precision.

\- Recall.

\- F1-score.



\#### 6.3 Transformer-Based Metadata Prediction



A Transformer architecture was trained using geometric information from ModaNet annotations.



The model predicts garment categories from extracted geometric features.



It explores the use of attention mechanisms for structured garment metadata.



Its inference results depend on the quality and ordering of the Mask R-CNN predictions. Therefore, the model is considered experimental.



\#### 6.4 Variational Autoencoder



A VAE was trained on Fashion-MNIST.



Its main components are:

\- Encoder.

\- Latent representation.

\- Reparameterization.

\- Decoder.



The encoder learns a probability distribution in latent space, and the decoder reconstructs or generates fashion images.



The best recorded validation loss was 238.4394 using BCE and KL divergence per image.



\#### 6.5 Generative Adversarial Network



The GAN consists of:



\- Generator: Produces synthetic fashion images.

\- Discriminator: Distinguishes generated images from real images.



The model was trained for ten epochs. Generated samples and training visualizations were saved for evaluation.



A formal quantitative image-quality metric has not yet been calculated.



\#### 6.6 ONNX Runtime and C++ Integration



The ResNet-18 model was exported to ONNX format.



ONNX Runtime was used for CPU inference in Python and C++.



CMake was used for the C++ build workflow. The C++ inference implementation was tested separately from the Streamlit application.



\---



\### 7. Experimental Results



\#### 7.1 Overall Model Performance



| Model | Task | Main Result |

|---|---|---:|

| ResNet-18 | Fashion classification | 92.39% accuracy |

| Transformer | Metadata classification | 85.26% accuracy |

| Mask R-CNN (Scratch) | Detection | 0.0068 Box AP50 |

| Mask R-CNN (Transfer) | Detection | 0.1985 Box AP50 |

| Mask R-CNN (Transfer) | Segmentation | 0.1372 Mask AP50 |

| VAE | Image generation | Best validation loss: 238.4394 |

| GAN | Image generation | 10 training epochs |



\#### 7.2 ResNet-18 Results



| Metric | Result |

|---|---:|

| Test accuracy | 92.39% |

| Macro F1-score | 92.33% |



The classifier achieved strong results on the Fashion-MNIST test set.



Performance varied across categories, with stronger results for classes such as Trouser, Bag, Sneaker, and Ankle boot. Shirt classification was comparatively more challenging.



\#### 7.3 Mask R-CNN Results



| Experiment | Box AP50 | Mask AP50 |

|---|---:|---:|

| Training from scratch | 0.0068 | 0.0019 |

| Transfer learning | 0.1985 | 0.1372 |



Transfer learning improved both detection and segmentation performance compared with training from scratch.



However, the results remain limited and require further improvement.



\#### 7.4 Transformer Results



| Metric | Result |

|---|---:|

| Test accuracy | 85.26% |

| Macro F1-score | 79.78% |



The model demonstrated the ability to learn garment-category relationships from geometric metadata.



Its results should not be interpreted as direct visual garment recognition.



\#### 7.5 Generative Model Results



The VAE and GAN successfully completed their training experiments.



Saved outputs include:

\- Generated VAE images.

\- VAE reconstructions.

\- GAN-generated image samples.

\- GAN loss curves.

\- GAN epoch-wise visualizations.



Formal perceptual quality evaluation remains future work.



\---



\### 8. ONNX Performance Benchmark



The ResNet-18 ONNX model was benchmarked using 1,000 Fashion-MNIST test images.



\#### 8.1 Python ONNX Benchmark



| Metric | Result |

|---|---:|

| Images tested | 1,000 |

| Average latency | 2.70 ms/image |

| Single-image throughput | 369.72 images/second |

| Batch size | 32 |

| Batch throughput | 676.59 images/second |

| Baseline process RAM | 107.56 MB |

| Peak process RAM | 133.01 MB |

| Additional RAM | 25.45 MB |



\#### 8.2 Python and C++ Comparison



Earlier benchmark results:



| Metric | Python | C++ |

|---|---:|---:|

| Average latency | 2.68 ms | 6.21 ms |

| Single-image throughput | 372.84 | 160.98 |

| Batch throughput | 653.42 | 395.06 |



Both implementations used ONNX Runtime and CPU inference.



These measurements are experimental and should not be treated as a controlled comparison across all environments.



\---



\### 9. Application Interface



The Streamlit application integrates the project's major components.



\#### Garment Detection

Allows users to upload images and view Mask R-CNN detections, segmentation visualizations, and experimental Transformer metadata predictions.



\#### Fashion Classification

Allows users to upload an image and view the ResNet-18 prediction and class probabilities.



The classifier was trained on 28 × 28 grayscale Fashion-MNIST images. Ordinary fashion photographs are outside its validated input domain.



\#### Generative Models

Displays VAE-generated images, reconstructions, GAN-generated samples, and training visualizations.



\#### Model Information

Presents model descriptions, evaluation results, and relevant limitations.



\---



\### 10. Limitations



The project has several limitations:



1\. Mask R-CNN was trained using a small pilot setup, limiting detection and segmentation performance.

2\. Transformer predictions depend on geometric metadata and Mask R-CNN outputs.

3\. ResNet-18 was trained on Fashion-MNIST and is not validated for general fashion photographs.

4\. GAN and VAE outputs have not undergone formal perceptual quality evaluation.

5\. The ONNX benchmark measures the model inference process rather than the complete application's resource requirements.

6\. Dataset image licensing requires separate verification.

7\. Real-world performance has not been established through large-scale deployment testing.



\---



\### 11. Future Scope



Possible improvements include:



\- Training Mask R-CNN with more images and additional epochs.

\- Improving garment segmentation accuracy.

\- Training a classification model on higher-resolution real-world fashion images.

\- Exploring visual Transformer architectures.

\- Adding garment attribute extraction, such as color, pattern, and sleeve type.

\- Evaluating generative outputs using suitable quantitative metrics.

\- Optimizing inference and measuring complete application resource usage.

\- Exploring virtual catalog search and recommendation features.

\- Deploying the application through a scalable cloud environment.



\---



\### 12. Conclusion



FashionVision AI demonstrates the integration of multiple machine-learning and deep-learning techniques into a single fashion-analysis prototype.



The project successfully implemented image classification, garment detection and segmentation experiments, geometric metadata prediction, generative models, and ONNX-based inference.



ResNet-18 achieved 92.39% test accuracy on Fashion-MNIST, while transfer learning improved Mask R-CNN performance compared with training from scratch.



The project also provided practical experience with dataset preparation, model training, evaluation, optimization, C++ integration, and application development.



Although some components require further improvement before real-world deployment, FashionVision AI provides a foundation for exploring AI-assisted fashion recognition and digital cataloging.



\---



\### 13. Project Files



Important project documentation:



\- `README.md` — Project overview and setup instructions.

\- `REQUIREMENTS.md` — Requirements and measurable acceptance criteria.

\- `DATASET\_CARD.md` — Dataset information and licensing considerations.

\- `MODEL\_CARD.md` — Model descriptions, metrics, and limitations.

\- `GO\_NO\_GO.md` — Deployment recommendation and resource assessment.

\- `FINAL\_REPORT.md` — Complete academic project report.



\*\*Final Project Status:\*\* Suitable for academic demonstration and controlled prototype use; not yet production-ready.

