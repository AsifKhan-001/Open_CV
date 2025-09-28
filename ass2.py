import cv2
print("you must remember for True is 1 and for False is 0")
a=int(input("Do you want to use the opencv to record the video :"))
if(a==0):
    print("See you later")
else:
    
    b=int(input("do you want to show or save the file , 0 for show and for save and 1 for save: "))

    if(b==1):
        camera=cv2.VideoCapture(0)
        frame_w=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec=cv2.VideoWriter_fourcc(*'XVID')
        record=cv2.VideoWriter("My ass2VIDEO3.avi",codec,60,(frame_w,frame_h))
        while True:
            ret,frame=camera.read()
            if ret==False:
                print("Little problem there!")
                break
            record.write(frame)
            cv2.imshow("Recording & Saving",frame)
            if(cv2.waitKey(1)&0XFF==ord('q')):
                print("your vido file has been completed")
                break
            
        camera.release()
        record.release()
        cv2.destroyAllWindows
    if(b==0):
        camera1=cv2.VideoCapture(0)
            
        while True:
            ret1,frame1=camera1.read()
            if not ret1:
                print("Little problem there!")
                break
            cv2.imshow("Recording...",frame1)
            if(cv2.waitKey(1)&0XFF==ord('q')):
                print("your vido file has been completed")
                break
        camera1.release()
            
        cv2.destroyAllWindows