
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN


image_path = 'people.jpg'  


detector = MTCNN()

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image.")
    exit()

print(f"Image shape: {image.shape}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = detector.detect_faces(image_rgb)

print(f"Faces detected: {len(results)}")

for res in results:
    x1, y1, width, height = res['box']
    x2, y2 = x1 + width, y1 + height
    confidence = res['confidence']

    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    # Put confidence text above the rectangle
    cv2.putText(image_rgb, f'Conf: {confidence:.2f}', (x1, y1 - 10),
                cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)

# Display the image with detected faces
plt.imshow(image_rgb)
plt.axis('off')  # Hide the axis
plt.show()
