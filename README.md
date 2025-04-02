# Intracranial Hemorrhage Detection from CT Scans

This Streamlit application utilizes a deep learning model to predict intracranial hemorrhage (ICH) from CT scan images. The model, "ich_detector.h5," is employed for analysis. The user-friendly interface allows users to upload CT scan images and receive instant predictions.

## Overview

- **Model**: Utilizes a pre-trained deep learning model for intracranial hemorrhage detection.
- **Labels**: The model classifies images into "No Hemorrhage" (0) and "Hemorrhage Detected" (1).
- **User Interface**: Created using Streamlit for easy interaction and quick analysis.

## Setup and Execution

### Prerequisites

- Ensure Python is installed on your system.
- Install required packages using the following command:

    ```bash
    pip install streamlit Pillow numpy tensorflow
    ```

### Execution

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your_username/ich-detection.git
    cd ich-detection
    ```

2. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

3. **Upload an Image:**
    - Use the file uploader to select a CT scan image (JPG, JPEG, or PNG).

4. **View Prediction:**
    - The processed image and prediction result will be displayed.
    - An error message indicates "Hemorrhage Detected," and success triggers celebratory balloons for "No Hemorrhage."
  
5. **Run the Application Online:**

    https://intracranial-hemorrhage-detector.streamlit.app/

