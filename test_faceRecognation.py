import dlib
import cv2
import pickle
import numpy as np

# Load the .pkl file and inspect its structure
with open("C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\trained_face_data.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the correct data based on the structure
known_face_encodings = np.array(data['embeddings'], dtype=np.float64)  # Adjust the key if necessary
known_face_labels = data['labels']

# Initialize Dlib's face detector and face recognizer
detector = dlib.get_frontal_face_detector()
recognizer = dlib.face_recognition_model_v1('C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('C:\\Users\\Lenovo R\\Desktop\\MTCNN & D-LIP\\D-Lib\\shape_predictor_68_face_landmarks.dat')

# Start the video capture
video_capture = cv2.VideoCapture(0)

def recognize_face(face_encoding, known_encodings, labels):
    face_encoding = np.array(face_encoding, dtype=np.float64)
    matches = np.linalg.norm(known_encodings - face_encoding, axis=1)
    min_distance_index = np.argmin(matches)
    if matches[min_distance_index] < 0.6:
        return labels[min_distance_index]
    return "Unknown"

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = shape_predictor(gray, face)
        face_encoding = np.array(recognizer.compute_face_descriptor(frame, landmarks), dtype=np.float64)

        name = recognize_face(face_encoding, known_face_encodings, known_face_labels)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
