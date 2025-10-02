import cv2
import numpy as np
image = cv2.imread("Open_CV/low_quality_images.png") #use another image for good visiualisation
sharpen_kernel=np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])
sharped=cv2.filter2D(image,-1,sharpen_kernel)
cv2.imshow("original image",image)
cv2.imshow("sharped",sharped)
cv2.waitKey(0)
cv2.destroyAllWindows