import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import os

# Load trained model
model = load_model("ich.h5", compile=False)

# Label map
labels = {0: 'No ICH', 1: 'ICH'}
positive_class = {'ICH'}

# Predict function
def processed_img(img_path):
    img = load_img(img_path, target_size=(128, 128, 3))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    y_class = np.argmax(prediction)
    res = labels[y_class]
    confidence = prediction[y_class]
    return res.capitalize(), confidence

# Streamlit App
def run():
    st.set_page_config(page_title="ICH Detection", layout="centered",page_icon="🧠")

    # === Sidebar ===
    with st.sidebar:
        # Title and short description
        st.title("🧠 ICH Detector")
        st.markdown("This tool uses deep learning to detect intracranial hemorrhage (ICH) from MRI brain scans. Upload an image to get instant classification results.")

        # Dropdown: Project Overview
        with st.expander("📘 Project Overview", expanded=False):
            st.markdown("""
- Classifies MRI brain scans to detect **Intracranial Hemorrhage (ICH)**
- Output: `ICH` or `No ICH`
- Developed using deep learning (CNN + Attention)
""")

        # Dropdown: Model Details
        with st.expander("🧪 Model Details", expanded=False):
            st.markdown("""
- 📐 **Architecture**: CNN with Attention  
- 🎯 **Accuracy**: 92%  
- 🖼️ **Input Size**: 128 × 128 × 3  
- 💾 **Model Format**: `.h5`
""")

        # Always visible: Options
        st.markdown("### ⚙️ Options")
        st.selectbox("📺 Display Mode", ["Basic", "Detailed"])
        show_confidence = st.selectbox("📈 Show Confidence Score?", ["Yes", "No"])

        # Developer info
        st.markdown("### 👨‍💻 Developer Info")
        st.markdown("""
**Akshwin T**  
📧 [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)  """)

    # === Title ===
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 ICH Detection from MRI Scans</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: grey;'>Upload an MRI image or use the sample image to detect intracranial hemorrhage</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # === Upload Section ===
    st.subheader("📤 Upload MRI Image")
    img_file = st.file_uploader("Upload a JPG, JPEG, or PNG image", type=['jpg', 'jpeg', 'png'])

    # === Use Sample Image Button ===
    use_sample = st.button("📷 Use Sample Image Instead")

    if img_file or use_sample:
        if use_sample:
            sample_path = "./upload_image/ich.png"
            if not os.path.exists(sample_path):
                st.error("❌ Sample image not found. Please ensure 'ich.png' is in the 'upload_image' folder.")
                return
            save_path = sample_path
            image_info = Image.open(sample_path)
            image_name = "Sample: ich.png"
        else:
            upload_dir = "./upload_image"
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, img_file.name)
            with open(save_path, "wb") as f:
                f.write(img_file.getbuffer())
            image_info = Image.open(save_path)
            image_name = img_file.name

        # === Image Preview ===
        st.markdown("### 🖼 Preview:")
        st.markdown(f"**🗂 File:** `{image_name}` | 📐 Size: `{image_info.size}` px")

        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_info, caption="MRI Image", use_container_width=True)

        st.markdown("---")

        # === Prediction ===
        with st.spinner("🔎 Predicting... Please wait..."):
            result, confidence = processed_img(save_path)

        st.markdown("### 🧪 **Prediction Result:**")
        if result in positive_class:
            st.error("🚨 **ICH DETECTED!**", icon="⚠️")
        else:
            st.success("✅ **No ICH Detected.**", icon="✅")

        if show_confidence == "Yes":
            st.info(f"📊 **Model Confidence**: `{confidence * 100:.2f}%`")

# Run the app
if __name__ == "__main__":
    run()