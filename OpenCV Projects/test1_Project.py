import cv2
import face_recognition
import csv
from datetime import datetime

# ----------------------------
# Create or open CSV file
# ----------------------------
csv_file = open("Open_CV/attendance_records.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

# Write header only if file is empty
csv_file.seek(0, 2)
if csv_file.tell() == 0:
    csv_writer.writerow(["Name", "Time", "Status"])

# ----------------------------
# Load Known Faces Safely
# ----------------------------
known_face_encodings = []
known_face_names = []

# Load Person 1
known_person1_image = face_recognition.load_image_file("Open_CV/AsifKhanFormalImage.jpg")
encodings1 = face_recognition.face_encodings(known_person1_image)
if encodings1:
    known_face_encodings.append(encodings1[0])
    known_face_names.append("Asif Khan")
else:
    print("⚠️ No face found in AsifKhanFormalImage.jpg. Try a clearer photo.")

# Load Person 2
known_person2_image = face_recognition.load_image_file("Open_CV/person1.jpg")
encodings2 = face_recognition.face_encodings(known_person2_image)
if encodings2:
    known_face_encodings.append(encodings2[0])
    known_face_names.append("Actor")
else:
    print("⚠️ No face found in person1.jpg. Try a clearer photo.")

recorded_names = set()

# ----------------------------
# Prepare Overlay
# ----------------------------
overlay_frame = cv2.imread("Open_CV/Overlay.jpg")
target_y_start, target_y_end = 130, 730
target_x_start, target_x_end = 25, 455
target_height = target_y_end - target_y_start
target_width = target_x_end - target_x_start

# ----------------------------
# Start Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to access webcam.")
        break

    frame_height, frame_width, _ = frame.shape
    crop_x_start = (frame_width - target_width) // 2
    crop_x_end = crop_x_start + target_width
    crop_y_start = (frame_height - target_height) // 2
    crop_y_end = crop_y_start + target_height

    cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Skip frame if crop size mismatched
    if cropped_frame.shape[0] != target_height or cropped_frame.shape[1] != target_width:
        print("⚠️ Cropped frame size mismatch — skipping frame")
        continue

    overlay_frame[target_y_start:target_y_end, target_x_start:target_x_end] = cropped_frame

    # Detect faces
    face_locations = face_recognition.face_locations(cropped_frame, model="hog")
    face_encodings = face_recognition.face_encodings(cropped_frame, face_locations)

    # If no faces, just show frame
    if not face_encodings:
        cv2.putText(overlay_frame, "No face detected", (600, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Attendance Reader", overlay_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ----------------------------
    # Compare and Mark Attendance
    # ----------------------------
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Mark attendance if not already marked
        if name != "Unknown Person" and name not in recorded_names:
            recorded_names.add(name)
            time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            status = "Present"
            csv_writer.writerow([name, time_now, status])
            csv_file.flush()

        # Map coordinates back to overlay frame
        overlay_top = top + target_y_start
        overlay_bottom = bottom + target_y_start
        overlay_right = right + target_x_start
        overlay_left = left + target_x_start

        # Draw rectangle and labels
        cv2.rectangle(overlay_frame, (overlay_left, overlay_top), (overlay_right, overlay_bottom), (0, 0, 255), 2)
        cv2.putText(overlay_frame, name, (overlay_left, overlay_top - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if name != "Unknown Person":
            cv2.putText(overlay_frame, "Present", (overlay_left, overlay_top - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ----------------------------
    # Display Final Frame
    # ----------------------------
    cv2.imshow("Attendance Reader", overlay_frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
csv_file.close()
cv2.destroyAllWindows()