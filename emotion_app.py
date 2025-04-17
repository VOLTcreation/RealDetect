import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# Load emotion labels from classes.json
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Get the emotion labels from the JSON file
emotion_labels = class_names.get("emotion_labels", [])

# Load the emotion detection model
model = tf.keras.models.load_model("emotion_detector_model.h5")

# Function to preprocess and predict emotion from image
def preprocess_and_predict(image: np.ndarray) -> tuple:
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected", 0

    # Focus on the first detected face
    x, y, w, h = faces[0]
    face_region = img_array[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (48, 48))  # Resize to model input size
    face_region = face_region.astype('float32') / 255  # Normalize the image
    face_region = np.expand_dims(face_region, axis=-1)  # Add the channel dimension
    face_region = np.expand_dims(face_region, axis=0)  # Add the batch dimension

    # Predict the emotion
    emotion_prediction = model.predict(face_region)
    predicted_class = np.argmax(emotion_prediction)
    confidence = emotion_prediction[0][predicted_class] * 100

    return emotion_labels[predicted_class], confidence

# Streamlit UI Setup
st.title("Real-Time Emotion Detection")

# Set up webcam capture
camera = st.camera_input("Capture video for emotion detection")

# Initialize a placeholder for real-time emotion detection results
stframe = st.empty()

if camera:
    video_capture = cv2.VideoCapture(camera)
    
    while True:
        ret, frame = video_capture.read()

        if not ret:
            st.warning("Failed to capture video frame.")
            break

        # Process frame and predict emotion
        emotion, confidence = preprocess_and_predict(frame)

        # Display the image with predicted emotion
        cv2.putText(frame, f"{emotion}: {confidence:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Real-Time Emotion Detection", frame)

        # Show frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Escape from the loop if the user presses 'q' on the webcam window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
