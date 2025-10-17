import cv2
import mediapipe as mp
import time
import numpy as np


cap=cv2.VideoCapture(0) # 0 for laptop camera but if here is 1 then its find the external source not operate on laptop camera and show error

mpHands = mp.solutions.hands
hands = mpHands.Hands() # in the func. we have 3 dta to pass static_image_mode=False,max_num_hands=2,,etc thats bydefault is good 
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0
thumbs_path = np.zeros((1080,1920,3),dtype= np.uint8) #its create a zero arrays according to our frame pixels , so here 480X640 are row and columnand 3 is channels like BGR
points= []   #its store the cordinates of the landmarks
while True:
    ret,frame=cap.read()
    h,w,_ = frame.shape #to check the frame height and width to put in thumbs_path 
    #print(h," X ",w)
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #mediapipe not support BGR so first we need to convert in RGB
    results=hands.process(frameRGB)


    #20 landmarks on the hand
    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            for id,lm in enumerate(handLMs.landmark): #this get unique id of each landmarks and the positions of hands components like thumb tip and palm etc. in x,y and z coordinates form 
                #print(id,lm) 
                #we wants the actual pixels
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h) #to required the centre position of the landmarks of the hands components ( each dat is the landmarks on the hand)
                print(id," : ",cx,cy) #we can store these datas in list and use for different purpose
                #Its help to track particular landmarks / portion of the hands
                #if we reduce if then its draw that circle for all but that no any actual mean
                if id==4: #id 4 is for our thumb top landmark and i think id 0 for just center of this all landmarks just below the palm
                    cv2.circle(frame,(cx,cy),15,(255,255,0),cv2.FILLED) # draw a filled circle and here 15 is the radius 
                    points.append((cx,cy)) #its store the cordinate of the seleted movement of the landmarks in each frame
                    if len(points) > 50: #its make empty our list after its length more than 50 these two below lines actually doing that
                        points.clear()
                        thumbs_path[:]=0

            mpDraw.draw_landmarks(frame,handLMs,mpHands.HAND_CONNECTIONS)
    if len(points) > 2:
        pts = np.array(points,np.int32) #you can also write the pts=np.array(points,np.int32).reshape((-1,1,2)) then you not need to write the nixt line line no 43
        pts = pts.reshape((-1,1,2))
        cv2.polylines(thumbs_path,[pts],False,(0,255,0),4)
    thumbs_path = cv2.addWeighted(thumbs_path,0.9,np.zeros_like(thumbs_path),0.1,0) #study on the addWeighted to know ehy 0.9 and 0.1 , actually 0.9 its a alpha , means reduce 90% by old frame , o.1 beta , 0 is gamma
    frame = cv2.addWeighted(frame,1,thumbs_path,0.3,0) # its continous add a curve line on the screeen to trace thumb, /// ALPHA is actually camera visibility, BETA is curve visibility and GAMMA is brightness offset ////  beta = 0.3 darker the visibility and 0.9 lighter
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
