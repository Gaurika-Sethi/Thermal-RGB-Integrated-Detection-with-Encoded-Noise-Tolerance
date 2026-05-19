import os
import sys


PROJECT_ROOT = "/content/drive/MyDrive/thermal_vision_project"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# =========================
# IMPORTS
# =========================

from src.inference.detector import ThermalDetector

from src.inference.router import DetectionRouter

from src.inference.postprocess import (
    postprocess_detections
)


# =========================
# PATHS
# =========================

THERMAL_MODEL_PATH = (
    f"{PROJECT_ROOT}/training_outputs/thermal_best.pt"
)

RGB_MODEL_PATH = (
    f"{PROJECT_ROOT}/training_outputs/rgb_yolov8s_v1-3/weights/best.pt"
)

THERMAL_IMAGE_DIR = (
    f"{PROJECT_ROOT}/datasets/LLVIP_YOLO/valid/images"
)

RGB_IMAGE_DIR = (
    f"{PROJECT_ROOT}/datasets/coco_subset/images/val"
)

# =========================
# LOAD DETECTORS
# =========================

thermal_detector = ThermalDetector(
    model_path=THERMAL_MODEL_PATH,
    confidence_threshold=0.4,
    max_det=20,
    image_size=640,
    modality="thermal"
)

rgb_detector = ThermalDetector(
    model_path=RGB_MODEL_PATH,
    confidence_threshold=0.4,
    max_det=20,
    image_size=640,
    modality="rgb"
)


# =========================
# ROUTER
# =========================

router = DetectionRouter(
    rgb_detector=rgb_detector,
    thermal_detector=thermal_detector
)


# =========================
# TEST IMAGES
# =========================

image_files = sorted(
    os.listdir(RGB_IMAGE_DIR)
)[:5]

# =========================
# LOOP THROUGH IMAGES
# =========================

for image_name in image_files:

    image_path = os.path.join(
        RGB_IMAGE_DIR,
        image_name
    )

    print("\n" + "=" * 60)
    print(f"IMAGE: {image_name}")
    print("=" * 60)

    # =====================
    # ROUTE INFERENCE
    # =====================

    detections = router.route(
        image_path=image_path,
        modality="rgb"
    )

    # =====================
    # POSTPROCESS
    # =====================

    processed = postprocess_detections(
        detections,
        confidence_threshold=0.4
    )

    # =====================
    # PRINT RESULTS
    # =====================

    if len(processed) == 0:
        print("No detections found.")
        continue

    for idx, item in enumerate(processed):

        print(f"\nDetection {idx + 1}")

        print(item)