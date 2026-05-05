from ultralytics import YOLO
import cv2

# Load RT-DETR model (this is NOT YOLO, just same API)
model = YOLO("rtdetr-l.pt")

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (640, 640))

results = model(img)

output = results[0].plot()

cv2.imwrite("rtdetr_output.jpg", output)
print("Saved output as rtdetr_output.jpg")