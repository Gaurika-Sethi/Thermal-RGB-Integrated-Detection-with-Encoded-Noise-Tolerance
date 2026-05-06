from ultralytics import YOLO
from utils import preprocess_thermal

import matplotlib.pyplot as plt
import cv2
import os

# -----------------------------------
# Load Model
# -----------------------------------
model = YOLO("../models/yolov8m.pt")

# -----------------------------------
# Thermal Folder
# -----------------------------------
thermal_folder = "../data/thermal"

# -----------------------------------
# Loop Through Images
# -----------------------------------
for file in os.listdir(thermal_folder):

    path = f"{thermal_folder}/{file}"

    # ONLY call preprocessing function
    img = preprocess_thermal(path)

    if img is None:
        print("Failed:", file)
        continue

    # -----------------------------------
    # Run YOLO
    # -----------------------------------
    results = model(img)

    print(f"\n{file}")
    print("Detections:", len(results[0].boxes))
    print("Confidences:", results[0].boxes.conf)

    # -----------------------------------
    # Draw detections
    # -----------------------------------
    output = results[0].plot()

    output = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------
    # Show result
    # -----------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(output)
    plt.title(f"Thermal YOLO: {file}")
    plt.axis("off")
    plt.show()