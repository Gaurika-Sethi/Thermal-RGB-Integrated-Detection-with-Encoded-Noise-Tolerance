import os
import sys
import json

PROJECT_ROOT = "/content/drive/MyDrive/thermal_vision_project"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.inference.detector import ThermalDetector
from src.reliability.features import extract_detection_features


# =========================
# PATHS
# =========================

MODEL_PATH = f"{PROJECT_ROOT}/training_outputs/thermal_best.pt"

IMAGE_DIR = f"{PROJECT_ROOT}/datasets/LLVIP_YOLO/valid/images"


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
# TEST FIRST 5 IMAGES
# =========================
all_records = []

image_files = sorted(os.listdir(IMAGE_DIR))[:5]


for image_name in image_files:


    image_path = os.path.join(IMAGE_DIR, image_name)

    detections = detector.predict(image_path)

    print("\n" + "=" * 60)
    print(f"IMAGE: {image_name}")
    print("=" * 60)

    if len(detections) == 0:
        print("No detections found.")
        continue

    for idx, detection in enumerate(detections):

        features = extract_detection_features(detection)
        features["image_name"] = image_name
        all_records.append(features)

        print(f"\nDetection {idx + 1}")
        print(features)

save_path = "/content/drive/MyDrive/thermal_vision_project/inference_outputs/thermal_metadata_v1/sample_detection_metadata.json"

with open(save_path, "w") as f:
    json.dump(all_records, f, indent=4)

print(f"\nSaved metadata to:\n{save_path}")