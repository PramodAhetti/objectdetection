import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to load and preprocess images
# def load_images(image_path, label):
#     images = []
#     labels = []
#     for img_file in os.listdir(image_path):
#         img = cv2.imread(os.path.join(image_path, img_file))
#         img = cv2.resize(img, (50, 50))  # Resize images to a consistent size
#         img = img / 255.0  # Normalize pixel values
#         images.append(img)
#         labels.append(label)
#     return images, labels

# # Load and preprocess data
# data = []

categories = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
# for i, category in enumerate(categories):
#     image_path = f'archive/data/{category}'
#     images, labels = load_images(image_path, i)
#     data.extend(zip(images, labels))

# # Shuffle the data
# np.random.shuffle(data)
# # s
# split_ratio = 0.8
# split_index = int(len(data) * split_ratio)
# train_data = data[:split_index]
# test_data = data[split_index:]

# train_images, train_labels = zip(*train_data)
# test_images, test_labels = zip(*test_data)

# train_images = np.array(train_images)
# test_images = np.array(test_images)
# train_labels = np.array(train_labels)
# test_labels = np.array(test_labels)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(8, activation='softmax')  # Using softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.load_weights('weights.h5')
# model.fit(train_images, train_labels, epochs=1, batch_size=10)
# model.save_weights('weights.h5')
# Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# Make predictions
test=cv2.imread('cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTA4L3Jhd3BpeGVsX29mZmljZV8xMl9waG90b19vZl9nb2xkZW5fcmV0cmlldmVyX3B1cHB5X2p1bXBpbmdfaXNvbF83MTM2NGE2OS1kZTM0LTQzMWEtYWRkZS04ZTdmZWQ0ZGFiOTIucG5n.png.webp');
test= cv2.resize(test, (50, 50)) # Resize images to a consistent size
test = test / 255.0
test=[test]
test=np.array(test);
predictions = model.predict(test)
# print(predictions)
predicted_class_index = np.argmax(predictions)
print(predicted_class_index)
# Map the predicted class index to the class name
predicted_class_name = categories[predicted_class_index]

# Print the predicted class name
print(f"Predicted Class: {predicted_class_name}")