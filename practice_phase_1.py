import cv2
#load the images
image = cv2.imread("/Users/asifkhan/Desktop/AI_ML_DATAsci./Open CV/free-youtube-logo-icon-2431-thumb.png")

print(image.shape)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
resize=cv2.resize(gray,(10,100))
cv2.imwrite("mytestimage.png",image)
cv2.imshow("practice_image😎",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

