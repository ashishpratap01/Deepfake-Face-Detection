import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from keras.models import load_model

# Load the trained model
model_path = 'video_classification_model1.h5'
model = load_model(model_path)

# Function to extract frames from the video
def extract_frames(video_path, n_frames=30, img_size=128):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_frames).astype(int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)

    cap.release()

    # Pad the frames if there are fewer than n_frames
    if len(frames) < n_frames:
        padding = [np.zeros((img_size, img_size, 3), dtype=np.uint8)] * (n_frames - len(frames))
        frames += padding

    frames = np.array(frames) / 255.0
    return np.expand_dims(frames, axis=0)

# Function to predict the video as REAL or FAKE
def predict_video(video_path):
    frames = extract_frames(video_path, n_frames=30, img_size=128)
    prediction = model.predict(frames)[0]
    real_prob = prediction[0]
    fake_prob = prediction[1]

    # If real or fake probability is greater than 70%, classify accordingly
    if real_prob >= 0.5:
        label = "REAL"
    elif fake_prob >= 0.5:
        label = "FAKE"
    else:
        label = "FAKE"  # Default to FAKE if neither meets the threshold

    return label, real_prob, fake_prob

# Streamlit User Interface
st.title("Deepfake Video Detection")
st.write("Upload a video (MP4, AVI, MOV) to detect if it's real or fake.")

# File uploader for video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyze"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.info("Processing video... please wait.")
        label, real_prob, fake_prob = predict_video(tmp_path)

        st.success(f"Prediction: **{label}**")
        #st.write(f"Class Probabilities → REAL: `{real_prob:.4f}`, FAKE: `{fake_prob:.4f}`")

        os.remove(tmp_path)
