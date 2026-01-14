import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Parameters
n_frames = 30
img_size = 128  # Match training input size
video_path = r"C:\Users\chand\OneDrive\Desktop\final_project_dataset\test_video.mp4"
model_path = 'video_classification_model1.h5'

# Load trained model
model = load_model(model_path)

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

    # Pad with black frames if the video is too short
    if len(frames) < n_frames:
        padding = [np.zeros((img_size, img_size, 3), dtype=np.uint8)] * (n_frames - len(frames))
        frames += padding

    # Normalize and return with batch dimension
    frames = np.array(frames) / 255.0
    return np.expand_dims(frames, axis=0)

# Extract frames from test video
video_frames = extract_frames(video_path, n_frames=n_frames, img_size=img_size)

# Predict
prediction = model.predict(video_frames)[0]
label_index = np.argmax(prediction)
label = "FAKE" if label_index == 1 else "REAL"
confidence = prediction[label_index]

# Output
print(f"Prediction: {label} (Confidence: {confidence:.4f})")
print(f"Class Probabilities -> REAL: {prediction[0]:.4f}, FAKE: {prediction[1]:.4f}")
