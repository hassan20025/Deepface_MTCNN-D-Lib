import os
import cv2
import numpy as np
import dlib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Initialize Dlib's models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\dlib_face_recognition_resnet_model_v1.dat')

# Set the path for dataset folder
dataset_path = 'C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\DATASET 2'
face_embeddings = []
face_labels = []

# Process Images to Generate Face Embeddings
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            faces = detector(image, 1)
            if len(faces) > 0:
                shape = shape_predictor(image, faces[0])
                embedding = np.array(face_recognition_model.compute_face_descriptor(image, shape))
                face_embeddings.append(embedding)
                face_labels.append(person_name)

# Filter out classes with fewer than 2 samples
face_embeddings = np.array(face_embeddings)
face_labels = np.array(face_labels)
label_counts = Counter(face_labels)
valid_labels = [label for label, count in label_counts.items() if count > 1]

filtered_embeddings = [embedding for embedding, label in zip(face_embeddings, face_labels) if label in valid_labels]
filtered_labels = [label for label in face_labels if label in valid_labels]

filtered_embeddings = np.array(filtered_embeddings)
filtered_labels = np.array(filtered_labels)

# Save filtered embeddings and labels to a .pkl file
with open('C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\trained_face_data.pkl', 'wb') as file:
    pickle.dump({'embeddings': filtered_embeddings, 'labels': filtered_labels}, file)

print("Trained face embeddings and labels have been saved to 'C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\trained_face_data.pkl'.")
