from ultralytics import YOLO
import cv2
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("Open_CV/shape.png",show=True)
cv2.waitKey(1)
cv2.destroyAllWindows()