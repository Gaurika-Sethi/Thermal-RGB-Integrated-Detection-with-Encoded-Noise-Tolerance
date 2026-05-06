from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# -----------------------------------
# Load Models
# -----------------------------------
yolo_model = YOLO("../models/yolov8n.pt")
rtdetr_model = YOLO("../models/rtdetr-l.pt")

# -----------------------------------
# Load Image
# -----------------------------------
img = cv2.imread("../data/rgb/sample.jpg")

if img is None:
    print("Image not found!")
    exit()

# Resize for consistency
img = cv2.resize(img, (640, 640))

# -----------------------------------
# Run YOLO
# -----------------------------------
yolo_results = yolo_model(img)

# -----------------------------------
# Run RT-DETR
# -----------------------------------
rtdetr_results = rtdetr_model(img)

# -----------------------------------
# Print Detection Counts
# -----------------------------------
print("YOLO detections:", len(yolo_results[0].boxes))
print("RT-DETR detections:", len(rtdetr_results[0].boxes))

# -----------------------------------
# Print Confidence Scores
# -----------------------------------
print("\nYOLO Confidences:")
print(yolo_results[0].boxes.conf)

print("\nRT-DETR Confidences:")
print(rtdetr_results[0].boxes.conf)

# -----------------------------------
# Draw Detection Outputs
# -----------------------------------
yolo_output = yolo_results[0].plot()
rtdetr_output = rtdetr_results[0].plot()

# Convert BGR → RGB for matplotlib
yolo_output = cv2.cvtColor(yolo_output, cv2.COLOR_BGR2RGB)
rtdetr_output = cv2.cvtColor(rtdetr_output, cv2.COLOR_BGR2RGB)

# -----------------------------------
# Show YOLO Output
# -----------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(yolo_output)
plt.title("YOLO Output")
plt.axis("off")
plt.show()

# -----------------------------------
# Show RT-DETR Output
# -----------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(rtdetr_output)
plt.title("RT-DETR Output")
plt.axis("off")
plt.show()

# -----------------------------------
# Save Outputs
# -----------------------------------
cv2.imwrite("../outputs/yolo_compare.jpg",
            cv2.cvtColor(yolo_output, cv2.COLOR_RGB2BGR))

cv2.imwrite("../outputs/rtdetr_compare.jpg",
            cv2.cvtColor(rtdetr_output, cv2.COLOR_RGB2BGR))

print("\nOutputs saved in outputs/ folder")