import cv2
camera=cv2.VideoCapture(0)
frame_width=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec=cv2.VideoWriter_fourcc(*'XVID')
recorder=cv2.VideoWriter("First_cap_Video.mp4",codec,30,(frame_width,frame_height))
while True:
    ret,frame=camera.read()
    if not ret:
        break
    recorder.write(frame)
    cv2.imshow("🎥Recoding...",frame)
    if cv2.waitKey(1)&0XFF==ord('q'):
        break
camera.release()
recorder.release()
cv2.destroyAllWindows()
