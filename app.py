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
tuberculosis = {'ICH'}

# Prediction function
def processed_img(img_path):
    img = load_img(img_path, target_size=(128, 128, 3))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    y_class = np.argmax(prediction)
    res = labels[y_class]
    confidence = prediction[y_class]
    return res.capitalize(), confidence

# Streamlit UI
def run():
    st.set_page_config(page_title="ICH Detection", layout="centered")
    st.title("🧠 Intracranial Hemorrhage Detection")
    st.subheader("Upload the MRI Image:")

    # Sidebar Info
    st.sidebar.header("About the Project 🧾")
    st.sidebar.markdown("📌 Uses CNN + Attention mechanism")
    st.sidebar.markdown("📌 Predicts whether ICH is present or not")
    st.sidebar.markdown("📌 Accuracy: **92%**")

    # Dropdowns
    display_mode = st.sidebar.selectbox("🔍 Display Mode", ["Basic", "Detailed"])
    show_confidence = st.sidebar.selectbox("📈 Show Confidence Score?", ["Yes", "No"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 Developed by Akshwin T")
    st.sidebar.markdown("📬 Contact: [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)")

    # File upload
    img_file = st.file_uploader("📤 Upload an MRI Image", type=['jpg', 'jpeg', 'png'])

    if img_file is not None:
        upload_dir = "./upload_image"
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, img_file.name)

        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Show image
        st.image(Image.open(save_path), caption='🖼 Uploaded Image', use_column_width=True)

        # Predict
        result, confidence = processed_img(save_path)
        if result in tuberculosis:
            st.error("🚨 **ICH DETECTED!**")
        else:
            st.success("✅ **No ICH Detected.**")

        # Optional confidence display
        if show_confidence == "Yes":
            st.markdown(f"**Confidence**: `{confidence * 100:.2f}%`")

        # Optional detailed info
        if display_mode == "Detailed":
            st.markdown("📚 *Model: Custom CNN with Attention*")
            st.markdown("🧪 *Input shape: (128, 128, 3)*")
            st.markdown("📊 *Model trained on annotated ICH dataset*")

# Run the app
if __name__ == "__main__":
    run()