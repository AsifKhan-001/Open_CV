import cv2
import face_recognition
import csv
from datetime import datetime


# To create a csv fileand store the data
csv_file = open("Open_CV/attendance_records.csv",mode="a",newline="")
csv_writer = csv.writer(csv_file)

#write header only if file is empty
csv_file.seek(0,2) #move to the end of file
if csv_file.tell()==0: #its check file empty or not
    csv_writer.writerow(["Name","Time","Status"])


#name = "Unknown Person"

#here i load known face encoding and more data
known_face_encodings = []
known_face_names = []

recorded_names = set() #make a set which name is recorded names

#Here i am load the known faces and there name 
known_person1_image = face_recognition.load_image_file("Open_CV/AsifKhanFormalImage.jpg") # similarly you can add so many persons and Try the image in jpg okk
known_person2_image = face_recognition.load_image_file("Open_CV/person1.jpg")

known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0] # similarly ad for each persons //// [0] we use the for select first face
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]      # face_encodings is make a list of numbers which length is 128 and which describe the face uniqueness.  /// if multiple face in one image this make multiple inner list of length 128.
#here i append the person1 encoding data in face encoding list
known_face_encodings.append(known_person1_encoding) # Add for each persons
known_face_encodings.append(known_person2_encoding)

#here i append the name in names list
known_face_names.append("Asif Khan") # Add for each persons.   //// 🛑 Must be arrange the name in same sequence of image
known_face_names.append("Actor")

#Now use your open cv knowledge


cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    #Here we find the face identity & locations in the current frame
    face_locations = face_recognition.face_locations(frame,model="hog")
    face_encodings = face_recognition.face_encodings(frame,face_locations)

    # This loop found the face in the frame through each faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings): #top → y-coordinate of the top edge of the face, right → x-coordinate of the right edge of the face, bottom → y-coordinate of the bottom edge of the face, left → x-coordinate of the left edge of the face.      ///// in the for loop you see top, bottom,left , right this data given by the locations function 
        # Checking the faces , if face matches any known faces
        face_width = right-left
        face_height = bottom-top
        if(face_width<300 or face_height<300):           #Here if the face is too far then the camera then our system not recoginise it ////.  if you want the student comes more near to camera then our system recognisie it then increase the value of 300 , but the 300 is good 150 not good
            continue
        matches= face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person" # if any unknown face detect then its show

        if True in matches:
            first_match_index = matches.index(True)          #Here first_match_index get a index from the matches
            name = known_face_names[first_match_index]       #here this indexx helps to find the correct name from the known_face_names
        
        if name != "Unknown Person" and name not in recorded_names: # these extra name not in record this stop to rewrite the same person data again and again
            recorded_names.add(name)
            time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            status = "Present"
            csv_writer.writerow([name,time_now,status])
            csv_file.flush() #record the all datas actually we use when some data stuck anywhere

        # Draw a box and labels
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-33),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        if name!= "Unknown Person":

            cv2.putText(frame,"present",(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    
    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
cap.release()
csv_file.close()
cv2.destroyAllWindows()