import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class VideoDataGenerator(Sequence):
    def __init__(self, video_dir, batch_size=32, img_size=(128, 128), n_frames=30, shuffle=True):
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_frames = n_frames
        self.shuffle = shuffle
        self.video_files = []

        # Collect all video file paths
        for category in os.listdir(video_dir):
            category_path = os.path.join(video_dir, category)
            if os.path.isdir(category_path):
                for video_file in os.listdir(category_path):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        self.video_files.append(os.path.join(category_path, video_file))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.video_files)

    def __getitem__(self, index):
        batch_videos = self.video_files[index * self.batch_size: (index + 1) * self.batch_size]
        frames = []
        labels = []

        for video_file in batch_videos:
            video = self._extract_frames(video_file)
            frames.append(video)

            # Case-insensitive label assignment
            label = 1 if "fake" in video_file.lower() else 0
            labels.append(label)

        return np.array(frames), np.array(labels)

    def _extract_frames(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frames = []
        count = 0

        while cap.isOpened() and count < self.n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
            count += 1

        cap.release()

        frames = np.array(frames, dtype=np.float32)

        # Pad if fewer frames than expected
        if len(frames) < self.n_frames:
            pad_shape = (self.n_frames - len(frames), *frames.shape[1:])
            padding = np.zeros(pad_shape, dtype=np.float32)
            frames = np.concatenate((frames, padding), axis=0)

        # Truncate if too many frames (extra safety)
        if frames.shape[0] > self.n_frames:
            frames = frames[:self.n_frames]

        # Normalize pixel values to [0, 1]
        frames /= 255.0
