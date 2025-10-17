import mediapipe as mp
import cv2

# Initialize Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            print("Face detected!")  # Print in terminal for test

    cv2.imshow("MediaPipe Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
