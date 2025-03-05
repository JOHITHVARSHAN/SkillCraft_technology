import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Path to the dataset
dataset_path = r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\train\train'
print("HI")

# Load images and labels
def load_data(dataset_path):
    images = []
    labels = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (64, 64))  # Resize images to 64x64
            images.append(image)
            if 'cat' in image_name.lower():
                labels.append('cat')
            elif 'dog' in image_name.lower():
                labels.append('dog')
    print("HI")
    return np.array(images), np.array(labels)

# Preprocess data
def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0  # Normalize pixel values
    images = images.reshape(images.shape[0], -1)  # Flatten images
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print("HI")
    return images, labels

# Load and preprocess data
images, labels = load_data(dataset_path)
if images.size == 0:
    raise ValueError("No images loaded. Please check the dataset path and ensure it contains images.")
images, labels = preprocess_data(images, labels)
print("HI")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print("HI")

# Make predictions
y_pred = svm.predict(X_test)
print("HI")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("HI")

# Function to predict whether an image is a cat or dog
def predict_image(image_path, model):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (64, 64))  # Resize image to 64x64
        image = image.astype('float32') / 255.0  # Normalize pixel values
        image = image.reshape(1, -1)  # Flatten image
        prediction = model.predict(image)
        if prediction == 0:
            return 'cat'
        else:
            return 'dog'
    else:
        return 'Invalid image path'

# Example usage
image_path = r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\test\cat.1.jpg'
result = predict_image(image_path, svm)
print(f'The image is a {result}')
