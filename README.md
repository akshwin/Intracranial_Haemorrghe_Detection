
# 🧠 ICH Detector – Intracranial Hemorrhage Classification App

An AI-powered web application to **detect Intracranial Hemorrhage (ICH)** from brain MRI scans using a deep learning model built with **CNN + Attention**. The app is built on **Streamlit** and designed for ease of use, fast diagnosis, and real-time inference.

<p align="center">
  <a href="https://intracranial-hemorrhage-detector.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20App-Click%20Here-blue?style=for-the-badge" alt="Live App Badge"/>
  </a>
</p>

---

## 📌 Table of Contents
- [🧠 Abstract](#-abstract)
- [🚀 Key Features](#-key-features)
- [📂 Dataset Used](#-dataset-used)
- [🧠 Methodology](#-methodology)
- [📈 Performance Metrics](#-performance-metrics)
- [⚙️ Installation & Usage](#️-installation--usage)
- [📊 Evaluation & Results](#-evaluation--results)
- [🛣 Future Scope](#-future-scope)
- [📜 License & 📧 Contact](#-license--contact)

---

## 🧠 Abstract

Intracranial Hemorrhage (ICH) is a serious medical condition requiring urgent diagnosis. This project presents an automated solution using a custom **Convolutional Neural Network with Attention mechanism** to detect ICH from brain MRI images. The model is deployed using Streamlit to enable quick access and seamless clinical decision support.

---

## 🚀 Key Features

- 📤 Upload MRI scans in JPG, JPEG, or PNG formats.
- 🧠 AI model classifies images as **ICH** or **No ICH**.
- 📊 Displays **model confidence score**.
- 🧼 Clean and interactive UI with **sidebar info**.
- ⚡ Real-time inference using `.h5` deep learning model.

---

## 📂 Dataset Used

- Publicly available brain MRI scan datasets were used.
- Images were resized to `128x128x3`.
- Labels: **ICH** and **No ICH**
- Data augmentation and preprocessing were applied for better generalization.

---

## 🧠 Methodology

```plaintext
[ Input MRI Image ]
        │
        ▼
[ Preprocessing (128x128 resize) ]
        │
        ▼
[ CNN + Attention Mechanism ]
        │
        ▼
[ Dense Layer + Softmax ]
        │
        ▼
[ Output: ICH or No ICH + Confidence ]
````

---

## 📈 Performance Metrics

| Metric               | Value |
| -------------------- | ----- |
| Accuracy             | 92.1% |
| Precision            | 90.4% |
| Recall (Sensitivity) | 91.8% |
| F1 Score             | 91.1% |

---

## ⚙️ Installation & Usage

### 🔧 Prerequisites

Make sure you have Python 3.x installed.

### 🔨 Installation Steps

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/ich-detector.git
cd ich-detector
````

```bash
# Step 2: (Optional) Create a virtual environment
python -m venv venv
# Activate:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

````
```bash
# Step 3: Install required packages
pip install -r requirements.txt
````
```bash
# Step 4: Run the Streamlit app
streamlit run app.py
```

Now, go to `http://localhost:8501` in your browser to access the app locally.

---

## 📊 Evaluation & Results

* Classification Report with high recall for ICH class.
* ROC-AUC \~ 0.95.
* Interactive predictions via live web demo.

---

## 🛣 Future Scope

* ✅ Improve model with larger and multi-modal datasets.
* ✅ Add Grad-CAM visualization for explainability.
* ✅ Expand classification into multi-class hemorrhage types.
* ✅ Add cloud-based backend for scalability.

---

## 📦 Requirements

Contents of `requirements.txt`:

```
numpy
pandas
matplotlib
keras
tensorflow
Pillow
ipython
```

---

## 📁 Project Structure

```
ich-detector/
├── app.py                # Streamlit App Code
├── ich.h5                # Trained model file
├── requirements.txt
├── README.md
└── upload_image/         # Temporary uploaded images
```

---

## 📜 License & 📧 Contact

**License**: MIT License – feel free to use and modify with attribution.
**Author**: Akshwin T
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/) | [GitHub](https://github.com/akshwin)

---

<p align="center">
  ⭐ If you found this project useful, consider giving it a star on GitHub!
</p>

---

