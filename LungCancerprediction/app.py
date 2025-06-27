import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import cv2

# Load trained model
model = tf.keras.models.load_model("LungCancerPrediction.h5")

# Class labels from your dataset
class_names = ['lung_acc', 'normal', 'lung_scc']

# Set up Streamlit UI
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("Lung Cancer Detection from Image")
st.write("Upload a chest X-ray or lung scan image to predict lung cancer type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def import_and_predict(image_data, model):
        img = np.array(image)
        img = cv2.resize(img, (256,256))
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)
        prediction=model.predict(img)
        return prediction

# If image uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Predicting..."):
        prediction = import_and_predict(image, model)
        index=np.argmax(prediction)
        predicted_class = class_names[index]

    st.success(f"Prediction:{predicted_class}")

else:
    st.warning("Please upload an image to continue.")
