from ultralytics import YOLO
import cv2

# Load models
yolo_model = YOLO("yolov8n.pt")
rtdetr_model = YOLO("rtdetr-l.pt")

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (640, 640))

# Run both
yolo_results = yolo_model(img)
rtdetr_results = rtdetr_model(img)

# Print detection counts
print("YOLO detections:", len(yolo_results[0].boxes))
print("RT-DETR detections:", len(rtdetr_results[0].boxes))

# Draw outputs
yolo_img = yolo_results[0].plot()
rtdetr_img = rtdetr_results[0].plot()

# Save
cv2.imwrite("yolo_output.jpg", yolo_img)
cv2.imwrite("rtdetr_output.jpg", rtdetr_img)

print("Both outputs saved")