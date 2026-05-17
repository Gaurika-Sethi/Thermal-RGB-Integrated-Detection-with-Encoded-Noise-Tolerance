from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Load Models
# -----------------------------------
yolo_model = YOLO("../models/yolov8m.pt")
rtdetr_model = YOLO("../models/rtdetr-l.pt")

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

    # -----------------------------------
    # Run YOLO
    # -----------------------------------
    yolo_results = yolo_model(img)

    # -----------------------------------
    # Run RT-DETR
    # -----------------------------------
    rtdetr_results = rtdetr_model(img)

    # -----------------------------------
    # Detection Counts
    # -----------------------------------
    print(f"\n===== {file} =====")

    print("\nYOLO detections:",
          len(yolo_results[0].boxes))

    print("RT-DETR detections:",
          len(rtdetr_results[0].boxes))

    # -----------------------------------
    # YOLO Confidences
    # -----------------------------------
    print("\nYOLO Confidences:")
    print(yolo_results[0].boxes.conf)

    # -----------------------------------
    # RT-DETR Confidences
    # -----------------------------------
    print("\nRT-DETR Confidences:")
    print(rtdetr_results[0].boxes.conf)

    # -----------------------------------
    # Draw YOLO output
    # -----------------------------------
    yolo_output = yolo_results[0].plot()

    # Convert BGR → RGB
    yolo_output = cv2.cvtColor(
        yolo_output,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------
    # Draw RT-DETR output
    # -----------------------------------
    rtdetr_output = rtdetr_results[0].plot()

    # Convert BGR → RGB
    rtdetr_output = cv2.cvtColor(
        rtdetr_output,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------
    # Show YOLO Result
    # -----------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(yolo_output)
    plt.title(f"YOLO: {file}")
    plt.axis("off")
    plt.show()

    # -----------------------------------
    # Show RT-DETR Result
    # -----------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(rtdetr_output)
    plt.title(f"RT-DETR: {file}")
    plt.axis("off")
    plt.show()