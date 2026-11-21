\# FashionVision AI — Dataset Card



\## 1. Overview



FashionVision AI uses two primary datasets: Fashion-MNIST and ModaNet. They support different machine learning tasks, including image classification, object detection, instance segmentation, metadata classification and image generation.



\---



\## 2. Fashion-MNIST



\### Source

\- Dataset: Fashion-MNIST

\- Publisher: Zalando Research

\- Repository: https://github.com/zalandoresearch/fashion-mnist

\- License: MIT License (dataset repository)



\### Dataset Description



Fashion-MNIST is a collection of grayscale fashion images designed for image classification research.



It contains:

\- 60,000 training images.

\- 10,000 test images.

\- 10 fashion categories.

\- Image dimensions of 28 × 28 pixels.



\### Categories



1\. T-shirt/top

2\. Trouser

3\. Pullover

4\. Dress

5\. Coat

6\. Sandal

7\. Shirt

8\. Sneaker

9\. Bag

10\. Ankle boot



\### Usage in This Project



Fashion-MNIST was used for:

\- ResNet-18 image classification.

\- Variational Autoencoder training.

\- Generative Adversarial Network training.



\### Preprocessing



\- Images resized or maintained at 28 × 28 pixels.

\- Grayscale image representation.

\- Pixel normalization to the range \[0, 1] for the relevant training pipeline.

\- Labels mapped to the corresponding fashion categories.



\---



\## 3. ModaNet



\### Source

\- Dataset: ModaNet

\- Research paper: ModaNet: A Large-Scale Street Fashion Dataset with Polygon Annotations.

\- Annotation repository: https://github.com/eBay/modanet

\- Annotation license: CC BY-NC 4.0, as specified by the annotation repository.

\- Image archive used: Community-hosted archive associated with cad0p/maskrcnn-modanet.



\### Dataset Description



ModaNet is a fashion dataset containing street-fashion images with polygon annotations.



The annotation file used in this project contains approximately:

\- 52,377 images.

\- 265,705 annotations.

\- 13 garment categories.



\### Categories



1\. Bag

2\. Belt

3\. Boots

4\. Footwear

5\. Outer

6\. Dress

7\. Sunglasses

8\. Pants

9\. Top

10\. Shorts

11\. Skirt

12\. Headwear

13\. Scarf/tie



\### Usage in This Project



ModaNet was used for:

\- Mask R-CNN garment detection.

\- Instance segmentation.

\- Fashion Transformer metadata classification.



\### Preprocessing



\- Annotation JSON processing.

\- Polygon annotation extraction.

\- Bounding-box generation.

\- Mask preparation.

\- Category ID mapping.

\- Geometric feature extraction for Transformer training.



\### Data Splitting



The Mask R-CNN dataset preparation used:

\- Training: 47,139 images.

\- Validation: 5,238 images.



The Transformer metadata dataset used:

\- Training: 41,803 samples.

\- Validation: 5,225 samples.

\- Testing: 5,226 samples.



The Transformer metadata preparation script uses random seed 42 for reproducibility.



\---



\## 4. Duplicate and Data Quality Checks



A formal image-level duplicate detection or perceptual-hashing audit has not yet been documented for either dataset.



Therefore, this project does not claim that all duplicate or near-duplicate images have been eliminated.



Future improvements should include:

\- Exact file-hash comparison.

\- Perceptual hashing for visually similar images.

\- Checking for overlapping images across training, validation and testing splits.

\- Verifying missing or corrupted image files.



\---



\## 5. Licensing and Usage Considerations



Fashion-MNIST is distributed under the MIT License through its dataset repository.



ModaNet's annotation repository specifies CC BY-NC 4.0 for its annotations. The image archive used in this project was community-hosted, so its contents should not automatically be assumed to share the annotation license.



Users should independently verify the rights and licensing conditions of the original images before redistribution or commercial use.



This project is intended for academic research, experimentation and educational demonstration.



\---



\## 6. Dataset Limitations



\### Fashion-MNIST

\- Contains small grayscale images.

\- Does not represent the full complexity of real-world fashion photographs.

\- Limited visual detail may affect generalization to real-world images.



\### ModaNet

\- Contains street-fashion images with variations in lighting, pose and occlusion.

\- Some garment categories have fewer examples than others.

\- Annotation quality and class imbalance may affect detection and segmentation performance.

\- Dataset licensing and image redistribution rights require careful consideration.



\---



\## 7. Intended Use



The datasets are used to demonstrate:

\- Fashion image classification.

\- Garment localization and segmentation.

\- Geometric garment metadata classification.

\- Fashion image generation.



The resulting models are experimental academic systems and should not be treated as fully validated commercial fashion-recognition services.



\---



\## 8. Conclusion



Fashion-MNIST and ModaNet provide complementary data for the FashionVision AI project. Fashion-MNIST supports classification and generative modelling, while ModaNet supports garment detection, segmentation and metadata learning.



Their different characteristics allow the project to demonstrate multiple approaches to fashion-focused artificial intelligence.

