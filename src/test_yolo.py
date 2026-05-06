from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Load YOLO Model
# -----------------------------------
model = YOLO("../models/yolov8m.pt")

# -----------------------------------
# RGB Folder
# -----------------------------------
rgb_folder = "../data/rgb"

# -----------------------------------
# Loop Through Images
# -----------------------------------
for file in os.listdir(rgb_folder):

    path = f"{rgb_folder}/{file}"

    img = cv2.imread(path)

    if img is None:
        print("Failed to load:", file)
        continue

    # Resize
    img = cv2.resize(img, (640, 640))

    # Run YOLO
    results = model(img)

    # Detection count
    print(f"\n{file}")
    print("Detections:", len(results[0].boxes))

    # Confidence scores
    print("Confidences:", results[0].boxes.conf)

    # Draw detections
    output = results[0].plot()

    # Convert BGR → RGB
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # Show result
    plt.figure(figsize=(8,8))
    plt.imshow(output)
    plt.title(f"YOLO Detection: {file}")
    plt.axis("off")
    plt.show()