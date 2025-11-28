# PSF-Aware Image Deblurring with CBAM-Infused Deep Residual Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A PSF-aware deep learning architecture for **document image deblurring**, built with a **4-channel input pipeline (RGB + PSF)**, enhanced **Residual Blocks**, and **CBAM Attention** to achieve high-quality restoration.

This repository includes the complete implementation inside:

📌 **`PSF_IMAGE_DEBLURRING_FINAL_MODEL.ipynb`**

---

## 🚀 Features

- **4-Channel Input:** Combines RGB + PSF into a single tensor.
- **Deep Residual CNN Architecture** enabling stable and efficient training.
- **CBAM Attention (Channel + Spatial)** integrated inside residual blocks.
- **Hybrid Loss Function:**  
  - MSE  
  - SSIM  
  - L1  
- **Evaluation Metrics:**  
  - PSNR  
  - SSIM  
- End-to-end pipeline:  
  ✔ Data loading  
  ✔ Model definition  
  ✔ Training & validation  
  ✔ Visualization  
  ✔ Inference

---


### 🗂 Dataset Used
**BMVC OCR Deblurring Dataset**

Each sample contains:
- `n_` folders store *blurred/noisy input images*.  
- `orig` stores *clean ground truth images*.  
- `psf` stores *kernel image*  

---

## 🧠 Model Architecture Overview

### **DeepDeBlurCNN**
Processes **4 input channels** (RGB + PSF). The key blocks include:

---

### 🔹 Initial Convolution Layer
Extracts low-level features.

---

### 🔹 Residual Blocks with CBAM Attention  

**Channel Attention Steps:**
1. **Squeeze:** Compress spatial dimensions.  
2. **Excitation:** MLP learns channel-wise weights.  
3. **Scaling:** Reweight input feature map.  
4. **Aggregation:** Pass enhanced features forward.  

**Spatial Attention:**  
Highlights important spatial regions using pooling + convolution.

---

### 🔹 Decoder / Upsampling Stages
Reconstruct the clean high-resolution image.

---

### 🔹 Final Convolution
Produces the restored **3-channel RGB output**.

---

## 🧪 Hybrid Loss Function


Why?
- **MSE:** Accurate pixel reconstruction  
- **SSIM:** Structural similarity + perceptual sharpness  
- **L1:** Edge and detail enhancement  

---

## ⚙ Training Configuration

| Parameter | Value |
|----------|-------|
| Epochs | 30 |
| Batch Size | 8 |
| Image Size | 224×224 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Scheduler | CosineAnnealingLR |
| Metrics | PSNR, SSIM |

---

## ▶ How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt

