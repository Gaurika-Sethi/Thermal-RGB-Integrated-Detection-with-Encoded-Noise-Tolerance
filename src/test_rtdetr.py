from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Load RT-DETR Model
# -----------------------------------
model = YOLO("../models/rtdetr-l.pt")

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
    # Run RT-DETR
    # -----------------------------------
    results = model(img)

    # -----------------------------------
    # Access detections
    # -----------------------------------
    boxes = results[0].boxes

    print(f"\n===== {file} =====")

    print("Detection Count:", len(boxes))

    # -----------------------------------
    # Bounding Boxes
    # -----------------------------------
    print("\nBounding Boxes (xyxy):")
    print(boxes.xyxy)

    # -----------------------------------
    # Confidence Scores
    # -----------------------------------
    print("\nConfidences:")
    print(boxes.conf)

    # -----------------------------------
    # Class IDs
    # -----------------------------------
    print("\nClass IDs:")
    print(boxes.cls)

    # -----------------------------------
    # Draw detections
    # -----------------------------------
    output = results[0].plot()

    # Convert BGR → RGB
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # -----------------------------------
    # Show output
    # -----------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(output)
    plt.title(f"RT-DETR: {file}")
    plt.axis("off")
    plt.show()