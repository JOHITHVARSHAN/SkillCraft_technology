from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from skimage import io, transform
import numpy as np
import os

# Define paths to your dataset
train_data_dir = 'path_to_train_data'
validation_data_dir = 'path_to_validation_data'
print("HI")

# Image dimensions
img_width, img_height = 150, 150

def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            img = io.imread(os.path.join(subdir, file))
            if img is not None:
                img = transform.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(os.path.basename(subdir))
    print("HI")
    return np.array(images), np.array(labels)

# Load and preprocess data
X_train, y_train = load_images_from_folder(train_data_dir)
X_val, y_val = load_images_from_folder(validation_data_dir)

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0

# Convert labels to one-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)

# Flatten images for MLP input
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# Building the model
model = MLPClassifier(hidden_layer_sizes=(512,), activation='relu', solver='adam', max_iter=50, verbose=True)

# Training the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation accuracy: {accuracy}')

# Save the model
import joblib
joblib.dump(model, 'hand_gesture_recognition_model.pkl')