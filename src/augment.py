import cv2
import numpy as np
import random


# =========================================================
# GAUSSIAN NOISE
# =========================================================

def gaussian_noise(image, severity=0.1):

    image = image.astype(np.float32) / 255.0

    noise = np.random.normal(
        loc=0,
        scale=severity,
        size=image.shape
    )

    corrupted = image + noise
    corrupted = np.clip(corrupted, 0, 1)

    return (corrupted * 255).astype(np.uint8)


# =========================================================
# CONTRAST
# =========================================================

def contrast(image, severity=0.1):

    factor = 1.0 + severity

    image = image.astype(np.float32)

    mean = np.mean(image)

    corrupted = (image - mean) * factor + mean
    corrupted = np.clip(corrupted, 0, 255)

    return corrupted.astype(np.uint8)


# =========================================================
# MOTION BLUR
# =========================================================

def motion_blur(image, severity=0.1):

    kernel_size = max(3, int(severity * 20))

    kernel = np.zeros((kernel_size, kernel_size))

    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    kernel = kernel / kernel_size

    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


# =========================================================
# SALT & PEPPER
# =========================================================

def salt_pepper(image, severity=0.1):

    corrupted = image.copy()

    num_pixels = int(severity * image.shape[0] * image.shape[1])

    # salt
    for _ in range(num_pixels):

        y = random.randint(0, image.shape[0] - 1)
        x = random.randint(0, image.shape[1] - 1)

        corrupted[y, x] = 255

    # pepper
    for _ in range(num_pixels):

        y = random.randint(0, image.shape[0] - 1)
        x = random.randint(0, image.shape[1] - 1)

        corrupted[y, x] = 0

    return corrupted


# =========================================================
# SENSOR DROPOUT
# =========================================================

def sensor_dropout(image, severity=0.1):

    corrupted = image.copy()

    h, w = image.shape[:2]

    dropout_w = int(w * severity)
    dropout_h = int(h * severity)

    x = random.randint(0, w - dropout_w)
    y = random.randint(0, h - dropout_h)

    corrupted[y:y+dropout_h, x:x+dropout_w] = 0

    return corrupted


# =========================================================
# APPLY CORRUPTION WRAPPER
# =========================================================

def apply_corruption(image, mode="gaussian_noise", severity=0.1):

    if mode == "gaussian_noise":
        return gaussian_noise(image, severity)

    elif mode == "contrast":
        return contrast(image, severity)

    elif mode == "motion_blur":
        return motion_blur(image, severity)

    elif mode == "salt_pepper":
        return salt_pepper(image, severity)

    elif mode == "sensor_dropout":
        return sensor_dropout(image, severity)

    else:
        raise ValueError(f"Unknown corruption mode: {mode}")