import os
import random
import cv2
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

DATASET_ROOT = "/content/drive/MyDrive/thermal_vision_project/datasets/coco_subset"

IMAGE_DIR = os.path.join(DATASET_ROOT, "images/train")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels/train")

NUM_SAMPLES = 20

# =========================
# GET RANDOM IMAGES
# =========================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

samples = random.sample(
    image_files,
    min(NUM_SAMPLES, len(image_files))
)

# =========================
# VISUALIZE
# =========================

for image_name in samples:

    image_path = os.path.join(IMAGE_DIR, image_name)

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h_img, w_img, _ = image.shape

    if os.path.exists(label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            cls, x, y, w, h = map(float, line.strip().split())

            # Convert YOLO → pixel coords
            x1 = int((x - w / 2) * w_img)
            y1 = int((y - h / 2) * h_img)

            x2 = int((x + w / 2) * w_img)
            y2 = int((y + h / 2) * h_img)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.title(image_name)
    plt.axis("off")
    plt.show()