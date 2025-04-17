import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import json

# Load emotion labels from classes.json
with open("classes.json", "r") as f:
    class_data = json.load(f)

# Get the emotion labels from the JSON file
emotion_labels = class_data.get("emotion_labels", [])

# Load the emotion detection model
model = tf.keras.models.load_model("emotion_detector_model.h5")

# Function to preprocess and predict emotion from image
def preprocess_and_predict(image: Image.Image) -> tuple:
    # Convert the image to numpy array
    img_array = np.array(image.convert('RGB'))  # Convert image to RGB format
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV compatibility
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect face using OpenCV's Haar cascades (for real-time emotion detection)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected", 0
    
    x, y, w, h = faces[0]
    face_region = gray[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (48, 48))  # Resize to fit model input size
    face_region = face_region.astype('float32') / 255  # Normalize the image
    face_region = np.expand_dims(face_region, axis=-1) 
    face_region = np.expand_dims(face_region, axis=0)

    # Predict the emotion
    emotion_prediction = model.predict(face_region)
    predicted_class = np.argmax(emotion_prediction)
    confidence = emotion_prediction[0][predicted_class] * 100

    return emotion_labels[predicted_class], confidence

# Streamlit UI Setup
st.title("Emotion Detector with Keras Model")

# Input options
input_method = st.radio("Choose Input Method", ("Take a Photo", "Upload an Image"), horizontal=True)

image_obj = None
if input_method == "Take a Photo":
    cam_img = st.camera_input("Capture image")
    if cam_img:
        image_obj = Image.open(cam_img)

elif input_method == "Upload an Image":
    uploaded_img = st.file_uploader("Upload an image for emotion detection", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image_obj = Image.open(uploaded_img)

if image_obj:
    st.image(image_obj, caption="📷 Your Image", use_container_width=True)

    with st.spinner("🔍 Detecting emotion..."):
        emotion, confidence = preprocess_and_predict(image_obj)

    st.success(f"Detected Emotion: **{emotion}** with **{confidence:.2f}%** confidence")

    if emotion == 'No face detected':
        st.warning("No face detected in the image. Please try another one.")
