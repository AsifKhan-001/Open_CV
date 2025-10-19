import cv2
import face_recognition

#here i load known face encoding and more data
known_face_encodings = []
known_face_names = []

#Here i am load the known faces and there name 
known_person1_image = face_recognition.load_image_file("Open_CV/AsifKhanFormalImage.jpg") # similarly you can add so many persons and Try the image in jpg okk


known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0] # similarly ad for each persons

#here i append the person1 encoding data in face encoding list
known_face_encodings.append(known_person1_encoding) # Add for each persons


#here i append the name in names list
known_face_names.append("Asif Khan")


#Now use your open cv knowledge


cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    #Here we find the face identity & locations in the current frame
    face_locations = face_recognition.face_locations(frame,model="hog")
    face_encodings = face_recognition.face_encodings(frame,face_locations)

    # This loop found the face in the frame through each faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Checking the faces , if face matches any known faces 
        matches= face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person" # if any unknown face detect then its show

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box and labels
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)


    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()








