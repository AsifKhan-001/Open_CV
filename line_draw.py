import cv2
image = cv2.imread("motu-patlu-images522.png")
if image is None:
    print(" sorry image could not found")
else:
    cv2.line(image,(20,120),(420,120),(10,100,50),6)
    cv2.line(image,(420,120),(420,300),(10,100,50),6)
    cv2.line(image,(420,300),(20,300),(10,100,50),6)
    cv2.line(image,(20,300),(20,120),(10,100,50),6)
    cv2.imshow("Line draw",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows