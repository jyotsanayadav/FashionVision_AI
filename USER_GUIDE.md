\# FASHIONVISION AI — USER GUIDE



\## 1. Introduction



FashionVision AI is a Streamlit-based application that demonstrates fashion image classification, garment detection, segmentation, metadata prediction, and image generation.



The application contains four main sections:



1\. Garment Detection

2\. Fashion Classification

3\. Generative Models

4\. Model Information



\## 2. System Requirements



Recommended environment:



\- Windows 10 or later.

\- Python installed.

\- Minimum 4 GB RAM recommended.

\- Internet connection for initial dependency installation.

\- Project files and trained model checkpoints available locally.



\## 3. Project Setup



\### Step 1: Open the Project Folder



Open PowerShell in the FashionVision\_AI project directory.



\### Step 2: Activate the Virtual Environment



Run:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### Step 3: Install Dependencies



If the dependencies have not already been installed, run:



```powershell

python -m pip install -r requirements.txt

```



\### Step 4: Launch the Application



Run:



```powershell

streamlit run app.py

```



The application will open in your default browser.



If it does not open automatically, use the local address displayed in the terminal.



\## 4. Using the Application



\### 4.1 Garment Detection



1\. Select \*\*Garment Detection\*\* from the sidebar.

2\. Upload a supported fashion image.

3\. Wait for the model to process the image.

4\. View the detected garment regions and segmentation visualization.

5\. Review the experimental Transformer metadata predictions when available.



\*\*Note:\*\* Detection and segmentation are experimental and may miss objects or predict incorrect boundaries.



\### 4.2 Fashion Classification



1\. Select \*\*Fashion Classification\*\*.

2\. Upload an image.

3\. Wait for ResNet-18 inference.

4\. View the predicted fashion category.

5\. Review the confidence score and class probability chart.



\*\*Important:\*\* The ResNet-18 model was trained on 28 × 28 grayscale Fashion-MNIST images. Its results on ordinary fashion photographs are not validated and should be treated as experimental.



\### 4.3 Generative Models



1\. Select \*\*Generative Models\*\* from the sidebar.

2\. Explore the VAE-generated images.

3\. View VAE reconstructions.

4\. Explore the GAN-generated samples.

5\. Review the GAN training visualizations and loss curve.



These are previously generated outputs saved in the project results directory.



\### 4.4 Model Information



1\. Select \*\*Model Information\*\*.

2\. Review the implemented models.

3\. Explore their evaluation metrics.

4\. Read the descriptions and limitations of each model.



\## 5. Important Project Directories



| Directory | Purpose |

|---|---|

| `app.py` | Main Streamlit application |

| `models/` | Trained model checkpoints and ONNX models |

| `dataset/` | Dataset files |

| `python/` | Python training and evaluation scripts |

| `src/` | C++ source files |

| `results/` | Evaluation results, generated images, and plots |

| `third\_party/` | External libraries and dependencies |

| `.streamlit/` | Streamlit configuration |



\## 6. Common Troubleshooting



\### Problem: Streamlit command not found



Activate the virtual environment:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Then run:



```powershell

python -m streamlit run app.py

```



\### Problem: Missing Python package



Install the missing package using:



```powershell

python -m pip install package\_name

```



Replace `package\_name` with the actual missing package.



\### Problem: Model file not found



Check whether the required model file exists inside the `models/` directory.



\### Problem: Application does not start



Check the terminal for the complete error message and verify that the virtual environment is active.



\### Problem: Application needs to be stopped



Press:



```text

Ctrl + C

```



in the terminal running Streamlit.



\## 7. Usage Recommendations



\- Use clear images with visible garments.

\- Avoid extremely blurry or low-quality images.

\- Allow the model to finish processing before changing pages.

\- Interpret experimental predictions carefully.

\- Do not treat model confidence as a guarantee of correctness.



\## 8. Conclusion



FashionVision AI provides an integrated interface for exploring fashion classification, object detection, segmentation, metadata prediction, and image generation.



This guide provides the basic instructions required to set up, launch, and explore the application.

