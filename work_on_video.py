import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if ret==False:
        print("counld not able to capture.")
        break
    cv2.imshow("your camera",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        print("Quitting......the whole")
        break
cap.release()
cv2.destroyAllWindows

