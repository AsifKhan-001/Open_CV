import cv2
import numpy as np
#import object_detection  #use python Object Detection 
from object_detection import ObjectDetection
od = ObjectDetection()
cap=cv2.VideoCapture("Open_CV/Los Angeles Highway Traffic.mp4")
while True:
    ret,frame=cap.read()

    #Object Detect in each Frame
    (class_ids,scores,boxes) = od.detect(frame) #this boxes actually stores the cordinates of the boxes over the object detection
    for box in boxes:
        (x,y,w,h) = box
        cv2.rectangle(frame,(x,y,(x+w,y+h)),(0,255,0),2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0XFF==ord("q"): #whenever i press "q" then its stop
        break
cap.release()
cv2.destroyAllWindows()