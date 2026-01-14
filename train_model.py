import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from video_data_generator import VideoDataGenerator


# Create a 3D CNN model
def create_model(input_shape=(30, 128, 128, 3), num_classes=2):
    model = models.Sequential()
    
    # 3D Conv Layers for video frame processing
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    
    # Flatten and add dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for classification

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Assuming you already have the directory path for the dataset
train_dir = r"C:\Users\chand\OneDrive\Desktop\final_project_dataset\split_dataset\train"
val_dir = r"C:\Users\chand\OneDrive\Desktop\final_project_dataset\split_dataset\val"

# Initialize the data generator
train_generator = VideoDataGenerator(train_dir, batch_size=32, img_size=(128, 128), n_frames=30, shuffle=True)
val_generator = VideoDataGenerator(val_dir, batch_size=32, img_size=(128, 128), n_frames=30, shuffle=False)

# Create the model
model = create_model(input_shape=(30, 128, 128, 3), num_classes=2)

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=10, verbose=1)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Save the trained model
model.save('video_classification_model2.h5')