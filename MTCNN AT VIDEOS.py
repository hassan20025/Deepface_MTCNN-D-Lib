import cv2
import mtcnn

faceDetector = mtcnn.MTCNN(min_face_size=50)

video_capture = cv2.VideoCapture('video.mp4')  

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read video frame.")
        break
    # Detect faces in the current frame
    result = faceDetector.detect_faces(frame)

    for res in result:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        confidence = res['confidence']
        key_points = res['keypoints'].values()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.putText(frame, f'conf: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

        for point in key_points:
            cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
