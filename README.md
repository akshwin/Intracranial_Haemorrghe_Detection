# 🧠 ICH Detector – Intracranial Hemorrhage Classification App

An AI-powered web application for detecting **Intracranial Hemorrhage (ICH)** from MRI brain scans. Built using **Streamlit**, powered by a custom **CNN with Attention** model, and optimized for real-time clinical inference.

🌐 **Live Demo:**  
[![Live Demo](https://img.shields.io/badge/Visit%20App-Click%20Here-blue?style=for-the-badge)](https://intracranial-hemorrhage-detector.streamlit.app/)

---

## 🚀 Features

- Upload MRI scans (JPG, JPEG, PNG).
- Deep learning model classifies the image as either **ICH** or **No ICH**.
- Option to view the **model confidence score**.
- Clean and interactive UI built using Streamlit.
- Sidebar with model overview and contact info.

---

## 🧠 Model Overview

- **Model Type:** CNN with Attention Mechanism  
- **Input:** MRI Brain Scan Image (128x128x3)  
- **Output:** Binary Classification – `ICH` or `No ICH`  
- **Activation:** Softmax (for confidence prediction)  
- **Performance:** ~92% Accuracy (on validation set)  
- **Frameworks:** Keras, TensorFlow  

---

## 🛠️ Tech Stack

- Python 3.x  
- Streamlit  
- TensorFlow / Keras  
- NumPy, Pillow  

---

## 📁 Project Structure

```

ich-detector/
├── app.py
├── ich.h5                  # Trained model file
├── requirements.txt
├── README.md
└── upload\_image/           # Temporary image storage (created dynamically)

````

---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ich-detector.git
cd ich-detector
````

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📥 Input Format

* Image files only: `.jpg`, `.jpeg`, `.png`
* Ensure the image is a **clear MRI scan** (not CT or X-ray).
* Preprocessed to 128x128 resolution.

---

## 📤 Output

* **ICH DETECTED**: Indicates signs of hemorrhage detected.
* **No ICH Detected**: Indicates a normal scan.
* **Model Confidence Score**: Optional, toggle from sidebar.

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

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👨‍💻 Author

Made with ❤️ by **Akshwin T**
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/) | [GitHub](https://github.com/akshwin)

---