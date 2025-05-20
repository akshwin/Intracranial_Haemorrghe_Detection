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
    st.set_page_config(page_title="ICH Detection", layout="centered")

    # === Sidebar ===
    with st.sidebar:
        st.markdown("## 🧾 **Project Overview**")
        st.markdown("""
        - 🤖 **Model**: CNN with Attention  
        - 🧠 **Task**: Intracranial Hemorrhage Detection  
        - 🎯 **Accuracy**: 92%  
        - 📚 **Input**: MRI Scan Image
        """)
        display_mode = st.selectbox("🔍 Display Mode", ["Basic", "Detailed"])
        show_confidence = st.selectbox("📈 Show Confidence Score?", ["Yes", "No"])

        st.markdown("---")
        st.markdown("👨‍💻 **Developed by:** Akshwin T")
        st.markdown("📬 [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)")

    # === Main Title ===
    st.markdown("<h1 style='text-align: center;'>🧠 ICH Detection from MRI Scans</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: grey;'>Upload a clear MRI image and get an instant diagnosis</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # === Upload Section ===
    with st.container():
        st.subheader("📤 Upload MRI Image:")
        img_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        if img_file:
            upload_dir = "./upload_image"
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, img_file.name)

            with open(save_path, "wb") as f:
                f.write(img_file.getbuffer())

            # Display image centered
            st.markdown("#### 🖼 Uploaded Image:")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(Image.open(save_path), width=300)

            # === Prediction ===
            result, confidence = processed_img(save_path)

            st.markdown("---")
            st.markdown("### 🔍 **Prediction Result:**")

            if result in positive_class:
                st.error("🚨 **ICH DETECTED!**")
            else:
                st.success("✅ **No ICH Detected.**")

            # Confidence Score
            if show_confidence == "Yes":
                st.info(f"**Model Confidence**: `{confidence * 100:.2f}%`")
# Run the app
if __name__ == "__main__":
    run()