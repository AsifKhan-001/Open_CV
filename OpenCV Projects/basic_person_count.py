import cv2
face_cascades=cv2.CascadeClassifier("Open_CV/haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec=cv2.VideoWriter_fourcc(*'XVID')
recorder=cv2.VideoWriter("Basic_Person_Count.mp4",codec,30,(frame_width,frame_height))
while True:
    ret,frame=cap.read()
    if not ret:
        print("Something went wrong")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascades.detectMultiScale(gray,1.3,3)
    
    for i,(x,y,w,h) in enumerate(faces,start=1):
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"PERSON {i} !",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,0,255),2)
    recorder.write(frame)
    cv2.imshow("Persons Counting",frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows




