# Multi-Organ Cell Instance Segmentation and Classification (Hover-Net)

This project implements a deep learning pipeline for simultaneous **instance segmentation** and **classification** of cells in multi-organ H&E stained tissue images. It leverages the **Hover-Net** architecture tailored with **ConvNeXt** backbone to handle touching nuclei and class imbalance.

## Performance

| Metric | Score |
| :--- | :--- |
| **Private Dataset Leaderboard Rank** | **2nd Place** |
| **Weighted Panoptic Quality (wPQ)** | **0.68** |

---

## Project Overview

The goal is to detect, segment, and classify individual cells into four categories:
1.  **Epithelial**
2.  **Lymphocyte**
3.  **Neutrophil**
4.  **Macrophage**

### Key Features
* **Architecture:** Modified Hover-Net using a pre-trained **ConvNeXt-Tiny** encoder and a custom Feature Pyramid Network (FPN) decoder.
* **Three-Branch Output:**
    * **NP (Nucleus Pixel):** Binary segmentation of nuclei.
    * **HV (Horizontal/Vertical):** Regression of distance maps to separate touching instances.
    * **TP (Type Pixel):** Pixel-wise classification maps.
* **Inference Pipeline:**
    * **Sliding Window Inference:** Handles large whole-slide images with overlap smoothing.
    * **Watershed Post-processing:** Uses distance transforms and peak detection to separate clustered cells.
    * **Grid Search:** Automates finding optimal post-processing thresholds.

---

## Methodology

### 1. Data Preprocessing

The notebook first converts bulky XML annotations into `.npz` files containing:

* `instance_map`: Unique ID for each cell instance.
* `type_map`: Class label for each cell instance.

### 2. Model Architecture (Hover-Net)

The model uses a **ConvNeXt-Tiny** backbone for feature extraction. The decoder upsamples features to generate three distinct maps:

1. **NP Map:** Predicts probability of a pixel being a nucleus.
2. **HV Map:** Predicts horizontal and vertical distances to the center of mass for each nucleus (crucial for separating touching cells).
3. **TP Map:** Predicts the cell type.

### 3. Training Strategy

* **Loss Function:** `2 * NP_Loss + 2 * HV_Loss + 1 * TP_Loss`
* `NP`: BCEWithLogitsLoss
* `HV`: MSELoss (masked to nucleus pixels)
* `TP`: **Focal Loss** to penalize misclassifying rare cells.

* **Augmentation:** Random flips, rotations, Gaussian blur, Cutout, and specialized H&E color jitter.

### 4. Post-Processing & Inference

Inference is performed using a **sliding window** approach. The resulting maps are processed as follows:

1. **NP Map** is thresholded to create a binary mask.
2. **HV Map** markers are extracted using `peak_local_max`.
3. **Watershed Algorithm** is applied to separate instances based on markers.
4. **Majority Voting** determines the class of each segmented instance using the **TP Map**.

---

## Evaluation Metric

The primary metric used is **Weighted Panoptic Quality (wPQ)**.

* **PQ** combines segmentation quality (IoU) and recognition quality (F1-score).
* **wPQ** averages the PQ scores of all classes, weighted by their importance (rare classes have higher weights).

```
wPQ = (Σ (PQ_c * weight_c)) / (Σ weight_c)
```