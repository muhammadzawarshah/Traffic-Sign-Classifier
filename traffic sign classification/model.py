import joblib
import cv2
import streamlit as st
import numpy as np
from skimage.feature import hog as skimage_hog
from skimage import exposure

# Load model and HOG parameters
model, hog_params = joblib.load("traffic_sign_model_with_params.pkl")

# OpenCV HOGDescriptor (for prediction)
hog = cv2.HOGDescriptor(
    _winSize=hog_params["winSize"],
    _blockSize=hog_params["blockSize"],
    _blockStride=hog_params["blockStride"],
    _cellSize=hog_params["cellSize"],
    _nbins=hog_params["nbins"]
)

# Streamlit UI
st.title("🚦 Traffic Sign Classifier (HOG + SVM)")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert uploaded file to numpy array (OpenCV format)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # BGR image

    # Preprocess for prediction
    resized = cv2.resize(image, hog_params["winSize"])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = hog.compute(gray).flatten().reshape(1, -1)

    # Prediction
    pred = model.predict(features)[0]
    st.success(f"Predicted Traffic Sign: **{pred}**")

    # Generate HOG visualization using skimage
    _, hog_image = skimage_hog(
        gray,
        orientations=hog_params["nbins"],
        pixels_per_cell=hog_params["cellSize"],
        cells_per_block=(
            hog_params["blockSize"][0] // hog_params["cellSize"][0],
            hog_params["blockSize"][1] // hog_params["cellSize"][1]
        ),
        visualize=True
    )

    # Normalize image for display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range='image', out_range=(0, 1))

    # Convert BGR → RGB for display
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Uploaded Image", width=250)
    with col2:
        st.image(hog_image_rescaled, caption="HOG Visualization", width=250)
