\# FashionVision AI — Go/No-Go Recommendation



\## 1. Executive Decision



\*\*Decision: GO for academic demonstration and controlled prototype use.\*\*



FashionVision AI successfully integrates fashion classification, garment detection, experimental metadata prediction, and generative models into a Streamlit application.



However, it is \*\*NOT recommended for production deployment\*\* because garment detection and segmentation performance still require improvement.



\## 2. Performance Summary



| Component | Evaluation Result | Decision |

|---|---:|---|

| ResNet-18 Classification | 92.39% accuracy, 92.33% macro F1 | GO |

| Mask R-CNN Detection | Box AP50: 19.85% | Improvement needed |

| Mask R-CNN Segmentation | Mask AP50: 13.72% | Improvement needed |

| Fashion Transformer | 85.26% accuracy, 79.78% macro F1 | Experimental |

| VAE | Successfully trained; generated samples available | GO for demonstration |

| GAN | 10 training epochs; generated samples available | GO for demonstration |



\## 3. Runtime and Resource Evaluation



The ResNet-18 ONNX model was benchmarked using CPU inference on 1,000 Fashion-MNIST test images.



| Metric | Measured Result |

|---|---:|

| Average inference latency | 2.70 ms/image |

| Single-image throughput | 369.72 images/second |

| Batch size | 32 |

| Batch throughput | 676.59 images/second |

| Baseline process RAM | 107.56 MB |

| Peak process RAM | 133.01 MB |

| Additional RAM usage | 25.45 MB |



These results indicate lightweight CPU inference for the tested ResNet-18 model. RAM figures represent the Python benchmark process, not the complete application's deployment requirements.



\## 4. Go/No-Go Assessment



\### GO

\- Academic demonstration.

\- Local execution through Streamlit.

\- Fashion-MNIST image classification.

\- Demonstration of VAE and GAN generated outputs.

\- Experimental garment detection and metadata prediction.



\### NO-GO for production deployment

\- Mask R-CNN detection and segmentation accuracy needs improvement.

\- Transformer predictions require further validation.

\- Classification on ordinary real-world fashion photographs is not validated.

\- Generative model quality needs formal quantitative evaluation.

\- Full application resource usage and deployment scalability have not been tested.



\## 5. Recommended Improvements



1\. Train Mask R-CNN on a larger dataset for more epochs.

2\. Improve garment detection and segmentation.

3\. Validate Transformer predictions against independent real-world samples.

4\. Evaluate GAN and VAE outputs using suitable quantitative metrics.

5\. Test the complete application under realistic deployment conditions.



\## 6. Final Recommendation



FashionVision AI is suitable for an academic prototype and project demonstration. Its ResNet-18 classifier demonstrates strong Fashion-MNIST performance and efficient CPU inference.



Further training, validation, and resource testing are required before considering production deployment.

