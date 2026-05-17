import cv2
import numpy as np
import random
from pathlib import Path


def gaussian_noise(image, severity=0.1):

    noise = np.random.normal(
        0,
        severity * 255,
        image.shape
    )

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def contrast_reduction(image, severity=0.5):

    mean = np.mean(image)

    reduced = (
        (image - mean) * severity
    ) + mean

    reduced = np.clip(reduced, 0, 255)

    return reduced.astype(np.uint8)


def salt_pepper(image, severity=0.02):

    corrupted = image.copy()

    num_pixels = int(severity * image.size)

    # Salt
    coords = [
        np.random.randint(0, i - 1, num_pixels)
        for i in image.shape[:2]
    ]

    corrupted[coords[0], coords[1]] = 255

    # Pepper
    coords = [
        np.random.randint(0, i - 1, num_pixels)
        for i in image.shape[:2]
    ]

    corrupted[coords[0], coords[1]] = 0

    return corrupted


def structured_patch(image, severity=0.3):

    corrupted = image.copy()

    h, w = image.shape[:2]

    patch_w = int(w * severity * 0.5)
    patch_h = int(h * severity * 0.5)

    x = random.randint(0, w - patch_w)
    y = random.randint(0, h - patch_h)

    corrupted[
        y:y+patch_h,
        x:x+patch_w
    ] = 0

    return corrupted


def sensor_dropout(image, severity=0.2):

    corrupted = image.copy()

    h, w = image.shape[:2]

    num_bands = max(1, int(severity * 4))

    for _ in range(num_bands):

        y = random.randint(0, h - 10)

        band_height = random.randint(5, 20)

        corrupted[
            y:y+band_height,
            :
        ] = 0

    return corrupted