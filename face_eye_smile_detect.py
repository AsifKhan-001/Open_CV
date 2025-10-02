import cv2
face_cascade=cv2.CascadeClassifier("Open_CV/haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("Open_CV/haarcascade_eye.xml")
smile_cascade=cv2.CascadeClassifier("Open_CV/haarcascade_smile.xml")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        roi_gray=gray[y:y+h,x:x+w] 
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,5)
        if len(eyes)>0:
            cv2.putText(frame,"EYES DETECTED",(x,y-50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        smiles=smile_cascade.detectMultiScale(roi_gray,1.7,20)
        if len(smiles)>0:
            cv2.putText(frame,"SMILE DETECTED",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),4)
    cv2.imshow("SMART FACE DETECTED",frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows