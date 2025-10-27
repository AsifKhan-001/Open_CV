#🛑 here i use dlib , imutils & scipy just to solve one problem that is the No add attendence on the bases of any image or photo , so i add here the blink of eye processur to make it faud or proxy proof


import cv2
import face_recognition
import dlib                    #its for the face detection and mainly for the face LANDMARKS which length is 68.
from imutils import face_utils      #we actually not use it you might be comment it or just ignore this warnings
from scipy.spatial import distance as dist
import csv
from datetime import datetime

from scipy.spatial import distance as dist           #this library helps me to distance between the landmarks , actually blink dection is detect when its measure the distance btw the two landmarks in each frame and if distance suddlenly decrease and then increase that's a blink

# Define a function to calculate the Eye Aspect Ratio (EAR) ///// its pretrained so not more focus on it
def eye_aspect_ratio(eye):      #EAR is detct the landmarks movement ,p0 and p3: left-most and right-most horizontal eye corners  &&&& p1,p2,p4,p5: vertical landmarks ///// its work like when vertical landmarks shrink then eyes close
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


predictor = dlib.shape_predictor("Open_CV/shape_predictor_68_face_landmarks.dat")     #this file download by other source and extract it then paste the path of that file which help to predictor  //////🛑 this gives the 68 landmarkes of the face these landmarks are include EYES,NOSE,EAR etc...
detector = dlib.get_frontal_face_detector()



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

blink_counter = 0

(leftEye, rightEye) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #with help of this function we specially decode the lendmarks of eyes and seperate it on the bases of left and right and hover on it to see the nose mouth just its give leftEye  = (42, 48) & rightEye = (36, 42) etc..            # its just define the lefteye and righteye here thats also pretrained but vs code not to catch it up so we actually wright in the code


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
            #we put a text that comes closer please
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         #convert to grayscale
        faces = detector(gray)                               #we detect the faces landmarks and more features with the help of DETECTOR
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)          #convert the dlib's landmarks into the numpy arry with the help of face_utils
            
            #Calculate EAR (eye open/close ratio)
            leftEyePts = shape[leftEye[0]:leftEye[1]]         #here we slicing the specific landmarks of shape which have the all landmarks of the face like leftear[0] stating landmark if left eye and lefteye[1] this is the ending landmarks of left eyes simlarly with right eye
            rightEyePts = shape[rightEye[0]:rightEye[1]]

            leftEAR = eye_aspect_ratio(leftEyePts)
            rightEAR = eye_aspect_ratio(rightEyePts)
            ear = (leftEAR + rightEAR) / 2.0

            #Check blink
            if ear < 0.20:                #0.20 is the thresold value of EAR
                blink_counter += 1
            else:
                if blink_counter >= 2:
                    print("Blink detected!")
                    blink_counter = 0
                    # only now mark attendance after the blink detected so the whole below code is under the this if condition
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