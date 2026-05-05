from ultralytics import YOLO
import cv2

# Load model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# Read image
img = cv2.imread("../data/sample.jpg")

# Resize (IMPORTANT — consistency)
img = cv2.resize(img, (640, 640))

# Run detection
results = model(img)

# Show result
output = results[0].plot()

cv2.imwrite("../outputs/yolo_output.jpg", output)
print("Saved output as yolo_output.jpg")
print(results[0].boxes)
print("Detections:", len(results[0].boxes))