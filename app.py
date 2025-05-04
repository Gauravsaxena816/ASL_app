import streamlit as st
import cv2
import numpy as np
import json
import time
import os
from tensorflow.keras.models import load_model
from preprocessing import process_frame  # Make sure this returns a preprocessed image

# Set up Streamlit page
st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("🤟 ASL Translator")

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'Models', 'asl_mediapipe_model.h5')
labels_path = os.path.join(BASE_DIR, 'Models', 'asl_class_indices_mediapipe.json')

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    model = load_model(model_path, compile=False)
    with open(labels_path, "r") as f:
        label_map = json.load(f)
    label_map = {v: k for k, v in label_map.items()}
    return model, label_map

model, label_map = load_model_and_labels()

# Initialize session state
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'prev_letter' not in st.session_state:
    st.session_state.prev_letter = ""
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0

# UI Layout
col1, col2 = st.columns([1, 2])
sentence_display = col2.empty()

# Sentence controls
with col2:
    col_space, col_reset = st.columns([1, 1])
    if col_space.button("Add Space"):
        st.session_state.sentence += " "
    if col_reset.button("Reset Sentence"):
        st.session_state.sentence = ""
        st.session_state.prev_letter = ""
        st.session_state.last_prediction_time = 0

# Capture image from browser
image_data = st.camera_input("📷 Show your ASL sign")

if image_data is not None:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    col1.image(frame, channels="BGR", caption="Captured Frame")

    # Preprocess and predict
    processed = process_frame(frame)

    if processed is not None:
        input_data = np.expand_dims(processed, axis=0)
        prediction = model.predict(input_data)[0]
        predicted_index = np.argmax(prediction)
        predicted_letter = label_map[predicted_index]
        confidence = prediction[predicted_index]
        current_time = time.time()

        if confidence > 0.8 and predicted_letter != st.session_state.prev_letter:
            if current_time - st.session_state.last_prediction_time > 1.5:
                st.session_state.sentence += predicted_letter
                st.session_state.prev_letter = predicted_letter
                st.session_state.last_prediction_time = current_time

        sentence_display.markdown(f"## ✍️ Sentence: `{st.session_state.sentence}`")
    else:
        st.warning("⚠️ Could not detect a valid hand region. Try again.")
