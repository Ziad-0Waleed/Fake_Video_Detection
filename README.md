# Deepfake Video Detection System

It is an end to end deep learning and software engineering solution designed to detect fake videos across various fabrications. This repository contains the core video deepfake detection pipeline, which utilizes a hybrid convolutional-recurrent neural network architecture optimized to detect subtle spatial and temporal manipulation artifacts in digital video.

## 🚀 Core Features

* **Hybrid Deep Learning Pipeline:** Combines advanced spatial feature extraction with sequential temporal modeling.
* **Optimized Preprocessing:** Automated aspect-ratio preservation via center-cropping, BGR-to-RGB color space alignment, and dynamic frame masking.
* **Production-Ready API:** Powered by FastAPI with automated request validation, structured error logging, and explicit 50MB file-size guardrails.
* **Cost-Sensitive Optimization:** Out-of-the-box resilience against heavily imbalanced datasets through integrated class-weight scaling.

## 🏗️ Model Architecture

The video classification framework is divided into two distinct processing phases, separating spatial awareness from temporal consistency analysis.

```text
[Raw Video Stream] 
       │
       ▼ (Sequential Frame Extraction)
[380x380 Tensors] 
       │
       ▼ (Spatial Feature Extraction)
[EfficientNetB4 Backbone (Frozen)] -> Extracts 1792-dim Feature Tensors
       │
       ▼ (Dynamic Mask Propagation)
[Stacked GRU Network] -------------> 64 Units -> 32 Units -> 16 Units
       │
       ▼ (Regularization Layers)
[Batch Norm & 0.5 Dropout]
       │
       ▼ (Classification Head)
[Dense Sigmoid Layer] -------------> Binary Classification Output (Real vs. Fake)
