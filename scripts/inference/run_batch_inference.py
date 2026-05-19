import os
import sys
import json
import cv2

from tqdm import tqdm


# =========================
# PROJECT ROOT
# =========================

PROJECT_ROOT = "/content/drive/MyDrive/thermal_vision_project"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# =========================
# IMPORTS
# =========================

from src.inference.detector import ThermalDetector
from src.reliability.features import extract_detection_features
from src.visualization.draw import draw_detections


# =========================
# PATHS
# =========================

MODEL_PATH = f"{PROJECT_ROOT}/training_outputs/thermal_best.pt"

IMAGE_DIR = f"{PROJECT_ROOT}/datasets/LLVIP_YOLO/valid/images"

OUTPUT_DIR = f"{PROJECT_ROOT}/inference_outputs/thermal_metadata_v1"

VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "batch_detection_metadata.json"
)


# =========================
# CREATE OUTPUT FOLDERS
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# =========================
# LOAD DETECTOR
# =========================

detector = ThermalDetector(
    model_path=MODEL_PATH,
    confidence_threshold=0.4,
    max_det=20,
    image_size=640,
    modality="thermal"
)


# =========================
# IMAGE LIST
# =========================

image_files = sorted(os.listdir(IMAGE_DIR))

# START SMALL FIRST
image_files = image_files[:100]


# =========================
# STORAGE
# =========================

all_records = []

summary = {
    "total_images": 0,
    "total_detections": 0,
    "empty_images": 0
}


# =========================
# RUN INFERENCE
# =========================

for image_name in tqdm(image_files):

    image_path = os.path.join(IMAGE_DIR, image_name)

    detections = detector.predict(image_path)

    summary["total_images"] += 1

    if len(detections) == 0:
        summary["empty_images"] += 1

    summary["total_detections"] += len(detections)


    # =====================
    # FEATURE EXTRACTION
    # =====================

    image_records = []

    for detection in detections:

        features = extract_detection_features(detection)

        features["image_name"] = image_name

        image_records.append(features)

        all_records.append(features)


    # =====================
    # SAVE VISUALIZATION
    # =====================

    vis_image = draw_detections(
        image_path,
        detections
    )

    save_vis_path = os.path.join(
        VIS_DIR,
        image_name
    )

    cv2.imwrite(save_vis_path, vis_image)


# =========================
# SAVE JSON METADATA
# =========================

with open(JSON_PATH, "w") as f:
    json.dump(all_records, f, indent=4)


# =========================
# PRINT SUMMARY
# =========================

print("\n" + "=" * 60)
print("BATCH INFERENCE SUMMARY")
print("=" * 60)

print(f"Total Images Processed : {summary['total_images']}")
print(f"Total Detections       : {summary['total_detections']}")
print(f"Images With No Dets    : {summary['empty_images']}")

print(f"\nMetadata Saved To:\n{JSON_PATH}")

print(f"\nVisualizations Saved To:\n{VIS_DIR}")