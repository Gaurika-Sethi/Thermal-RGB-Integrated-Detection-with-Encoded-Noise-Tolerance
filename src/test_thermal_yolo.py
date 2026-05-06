from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Load YOLO Model
# -----------------------------------
model = YOLO("../models/yolov8m.pt")

# -----------------------------------
# Thermal Folder
# -----------------------------------
thermal_folder = "../data/thermal"

# -----------------------------------
# Loop Through Thermal Images
# -----------------------------------
for file in os.listdir(thermal_folder):

    path = f"{thermal_folder}/{file}"

    # Load grayscale thermal image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load:", file)
        continue

    # -----------------------------------
    # CLAHE
    # -----------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img = clahe.apply(img)

    # -----------------------------------
    # Resize
    # -----------------------------------
    img = cv2.resize(img, (640, 640))

    # -----------------------------------
    # Convert to 3-channel
    # -----------------------------------
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # -----------------------------------
    # Run YOLO
    # -----------------------------------
    results = model(img)

    # -----------------------------------
    # Detection Info
    # -----------------------------------
    print(f"\n{file}")
    print("Detections:", len(results[0].boxes))
    print("Confidences:", results[0].boxes.conf)

    # -----------------------------------
    # Draw Detections
    # -----------------------------------
    output = results[0].plot()

    # Convert for matplotlib
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # -----------------------------------
    # Show Result
    # -----------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(output)
    plt.title(f"Thermal YOLO: {file}")
    plt.axis("off")
    plt.show()