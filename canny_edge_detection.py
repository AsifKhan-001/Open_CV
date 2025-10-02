import cv2
image = cv2.imread("Open_CV/Small Face Hairstyles copy.jpeg",cv2.IMREAD_GRAYSCALE)
edge = cv2.Canny(image,50,150)
cv2.imshow("ORIGINAL_IMG",image)
cv2.imshow("EDGE_IMG",edge)
cv2.waitKey(0)
cv2.destroyAllWindows