import cv2
image=cv2.imread("/Users/asifkhan/Desktop/AI_ML_DATAsci./OpenCV.png")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
flag=int(input("Do you want to show the image or not? "))
if(flag==1):
    cv2.imshow("YOUR IMAGE",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cv2.imwrite("opencvgray.png",gray)
    print("your image successfully saved")