# FaceMetrics: Detection-of-Age-and-Gender-Using-Deep-Learning

**High-Performance Demographic Profiling with MediaPipe**

This project delivers a lightweight, optimized system for detecting age and gender in real-time. Built upon Google's **MediaPipe** framework, it ensures low-latency performance suitable for edge devices and live video applications while maintaining high predictive accuracy.

---

## 🚀 Core Capabilities

* **Instant Analysis:** Leverages MediaPipe's highly optimized pipeline for millisecond-latency inference.
* **Dual-Model Architecture:**
    * **Age Estimator:** A regression-based deep learning model providing continuous age value estimation across all demographics.
    * **Gender Classifier:** A binary CNN optimized for robust classification across diverse lighting, angles, and ethnicities.
* **Edge-Ready:** Designed to run efficiently on CPUs and low-resource hardware without requiring heavy GPU clusters.

---

## ⚙️ Technical Architecture

The system operates on a three-stage pipeline:

1.  **Face Localization:** Uses MediaPipe Face Detection to instantly identify and crop regions of interest.
2.  **Preprocessing:** Normalization and resizing of facial crops for neural network compatibility.
3.  **Inference Engine:**
    * *Age Model:* Regression network minimizing Mean Absolute Error (MAE) (~±3 years).
    * *Gender Model:* Binary classifier achieving ~95% accuracy on test sets.

---

## 📂 Project Structure

```text
FaceMetrics/
├── models/             # Pre-trained .pb / .tflite inference models
├── src/
│   ├── detect.py       # Core logic for real-time inference
│   ├── utils.py        # Visualization and image pre-processing helpers
│   └── config.py       # System parameters and thresholds
└── README.md           # Documentation
