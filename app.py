import streamlit as st 
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np 
from keras.models import load_model 
import os

# Load Model
model = load_model("ich.h5")
labels = {0: 'No ICH', 1: 'ICH'}
ich = {'ICH'}

# Ensure Upload Directory Exists
UPLOAD_FOLDER = "./upload_image"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Image Processing Function
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Removed third dimension
    img = img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions correctly
    
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)[0]  # Get the class index
    res = labels[y_class]  # Map index to label
    
    return res.capitalize()

# Streamlit UI
def run():
    st.title("Intracranial Hemorrhage Detection 🧠")
    st.subheader("Upload the MRI Image:")

    st.sidebar.header("About the project:")
    st.sidebar.write("📌 The project is developed using a Convolutional Neural Network with an Attention mechanism.")
    st.sidebar.write("📌 The model detects whether the patient has Intracranial Hemorrhage or not.")
    st.sidebar.write("📌 The model achieved an accuracy of 92 percent.")

    img_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if img_file:
        img = Image.open(img_file).resize((250, 250))
        st.image(img)

        save_image_path = os.path.join(UPLOAD_FOLDER, img_file.name)
        
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = processed_img(save_image_path)

        if result in ich:
            st.error('**ICH DETECTED!!**')
        else:
            st.success('**NO ICH!!**')

# Run the Streamlit app
if __name__ == "__main__":
    run()
