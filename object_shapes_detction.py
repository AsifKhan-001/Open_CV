import cv2
image=cv2.imread("Open_CV/circle.png")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
contours,heirarcy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,0,255),3)
for contour in contours:
    approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    corners=len(approx)
    if corners==3:
        shape_name="TRIANGLE"
    elif corners==4:
        shape_name="RECTANGLE"
    elif corners==5:
        shape_name="PENTAGON"
    elif corners>10:
        shape_name="CIRCLE"
    else:
        shape_name="UNKNOWN"
    
cv2.drawContours(image,[approx],0,(255,0,0),2)
x=approx.ravel()[0]
y=approx.ravel()[1]-10
cv2.putText(image,shape_name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,0),2)
cv2.imshow("contours",image)
cv2.waitKey(0)
cv2.destroyAllWindows