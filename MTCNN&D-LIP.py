import cv2
import mtcnn

faceDetector = mtcnn.MTCNN(min_face_size=1)
image = cv2.imread('7.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not read image.")
    exit()

# Print the shape of the image
print(f"Image shape: {image.shape}")

# Detect faces
result = faceDetector.detect_faces(image)
print(f"Faces detected: {len(result)}")  # Check number of faces detected

# Draw rectangles and key points if faces are detected
for res in result:
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    confidence = res['confidence']
    key_points = res['keypoints'].values()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    cv2.putText(image, f'conf: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

    for point in key_points:
        cv2.circle(image, point, 5, (0, 255, 0), thickness=-1)

# Display the image with detected faces
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
