import cv2
import numpy as np


#Gussian_Blur & Median Blur
image = cv2.imread("Open_CV/nature.jpg")
g_blurred = cv2.GaussianBlur(image,(21,21),0)
m_blurred = cv2.medianBlur(image,9) 
cv2.imshow("Original image",image)
cv2.imshow("G_Blur image",g_blurred)
cv2.imshow("M_Blur",m_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows
