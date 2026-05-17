import os
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# Folder Paths
# -----------------------------
rgb_folder = "../data/rgb"
thermal_folder = "../data/thermal"

# =====================================================
# RGB LOOP
# =====================================================

for file in os.listdir(rgb_folder):

    path = f"{rgb_folder}/{file}"

    img = cv2.imread(path)

    if img is None:
        print("Failed to load RGB:", file)
        continue

    # Resize
    img = cv2.resize(img, (640, 640))

    # Print shape
    print(f"RGB -> {file} shape:", img.shape)

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image
    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    plt.title(f"RGB: {file}")
    plt.axis("off")
    plt.show()

# =====================================================
# THERMAL LOOP
# =====================================================

for file in os.listdir(thermal_folder):

    path = f"{thermal_folder}/{file}"

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load Thermal:", file)
        continue

    # -----------------------------
    # BEFORE CLAHE
    # -----------------------------
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title(f"Before CLAHE: {file}")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # Apply CLAHE
    # -----------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img = clahe.apply(img)

    # Resize
    img = cv2.resize(img, (640, 640))

    # Convert to 3-channel
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Print shape
    print(f"Thermal -> {file} shape:", img_color.shape)

    # Convert for matplotlib
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # AFTER CLAHE
    # -----------------------------
    plt.figure(figsize=(5,5))
    plt.imshow(img_color)
    plt.title(f"After CLAHE: {file}")
    plt.axis("off")
    plt.show()