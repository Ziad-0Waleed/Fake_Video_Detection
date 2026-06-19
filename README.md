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

```


### 1. Spatial Feature Extractor
* **Backbone:** EfficientNetB4 (pre-trained on ImageNet, classification head removed).
* **Input Resolution:** **380×380 pixels**, selected as the mathematical sweet spot to preserve fine-grained blending anomalies and edge inconsistencies around facial boundaries without causing memory overhead.
* **Output:** Global Average Pooling compresses spatial dimensions into a dense **1792-dimensional vector** per frame.

### 2. Temporal Sequence Modeler
* **Architecture:** Stacked Gated Recurrent Unit (GRU) network (64 → 32 → 16 units).
* **Masking Protocol:** Parallel boolean attention masks (`frame_masks`) track sequence boundaries, forcing downstream layers to disregard zero-padding elements at truncated or shorter timesteps.
* **Regularization:** Interspersed Batch Normalization standardizes intermediate activations, while dual **0.5 Dropout** layers mitigate overfitting before the classification boundary.
* **Output Head:** A single-unit Dense layer with Sigmoid activation mapping outputs to a raw probability score [0.0, 1.0].

## 📊 Dataset & Preprocessing Pipeline
The model is trained and evaluated using the Deepfake Detection Challenge (DFDC) dataset framework.

* **Geometry Correction:** Videos are dynamically sliced using a customized centering calculation to prevent geometric squishing or aspect ratio distortion.
* **Sequential Batching:** Frame rates are sub-sampled and bound to a strict maximum sequence length threshold to create uniform rectangular tensors.
* **Imbalance Mitigation:** To address the severe scarcity of real media samples, custom cost-sensitive loss weights are injected during compilation, penalizing the misclassification of genuine assets approximately four times harder than fake assets.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* CUDA-capable GPU (Recommended for extraction and training)

### Local Deployment

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
