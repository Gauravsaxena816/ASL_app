import streamlit as st
import cv2
import numpy as np
import json
import time
import os
from tensorflow.keras.models import load_model
from preprocessing import process_frame  # Ensure preprocessing.py is in the same directory

# Set up the Streamlit page
st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("ASL Translator")

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to the model and label files
model_path = os.path.join(BASE_DIR, 'Models', 'asl_mediapipe_model.h5')
labels_path = os.path.join(BASE_DIR, 'Models', 'asl_class_indices_mediapipe.json')

# Load the trained model and label map
@st.cache_resource
def load_model_and_labels():
    model = load_model(model_path, compile=False)
    with open(labels_path, "r") as f:
        label_map = json.load(f)
    # Reverse the label map: index -> label
    label_map = {v: k for k, v in label_map.items()}
    return model, label_map

model, label_map = load_model_and_labels()

# Initialize session state for sentence construction
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'prev_letter' not in st.session_state:
    st.session_state.prev_letter = ""
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0

# Initialize video capture
cap = cv2.VideoCapture(0)

# Layout for displaying video and sentence
col1, col2 = st.columns([1, 2])
video_display = col1.empty()
sentence_display = col2.empty()

# Buttons for sentence manipulation
with col2:
    col_space, col_reset = st.columns([1, 1])
    if col_space.button("Add Space"):
        st.session_state.sentence += " "
    if col_reset.button("Reset Sentence"):
        st.session_state.sentence = ""
        st.session_state.prev_letter = ""
        st.session_state.last_prediction_time = 0

# Parameters for prediction
time_threshold = 1.5  # seconds
confidence_threshold = 0.8  # adjust based on model performance

# Main loop for real-time prediction
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture video.")
        break

    # Process the frame to detect and crop the hand
    processed = process_frame(frame)
    if processed is None:
        video_display.image(frame, channels="BGR")
        continue

    # Expand dimensions and predict the letter
    input_data = np.expand_dims(processed, axis=0)
    prediction = model.predict(input_data)[0]
    predicted_index = np.argmax(prediction)
    predicted_letter = label_map[predicted_index]
    confidence = prediction[predicted_index]
    current_time = time.time()

    # Append the predicted letter to the sentence if conditions are met
    if confidence > confidence_threshold and predicted_letter != st.session_state.prev_letter:
        if current_time - st.session_state.last_prediction_time > time_threshold:
            st.session_state.sentence += predicted_letter
            st.session_state.prev_letter = predicted_letter
            st.session_state.last_prediction_time = current_time

    # Display the video feed and the constructed sentence
    video_display.image(frame, channels="BGR")
    sentence_display.markdown(f"## ✍️ Sentence: `{st.session_state.sentence}`")

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
