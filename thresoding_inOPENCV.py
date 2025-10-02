import cv2
image = cv2.imread("Open_CV/flower1.jpg",cv2.IMREAD_GRAYSCALE)
ret,thresold_img = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
cv2.imshow("original_img.......",image)
cv2.imshow("thresold_image",thresold_img)
cv2.waitKey(0)
cv2.destroyAllWindows