import cv2

# -----------------------------------
# RGB Preprocessing
# -----------------------------------
def preprocess_rgb(path):

    img = cv2.imread(path)

    if img is None:
        return None

    img = cv2.resize(img, (640, 640))

    return img


# -----------------------------------
# Thermal Preprocessing
# -----------------------------------
def preprocess_thermal(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img = clahe.apply(img)

    # Resize
    img = cv2.resize(img, (640, 640))

    # Convert to 3-channel
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img