import os
import sys

PROJECT_ROOT = "/content/drive/MyDrive/thermal_vision_project"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.inference.detector import ThermalDetector


MODEL_PATH = f"{PROJECT_ROOT}/training_outputs/thermal_best.pt"

IMAGE_DIR = f"{PROJECT_ROOT}/datasets/LLVIP_YOLO/valid/images"


detector = ThermalDetector(
    model_path=MODEL_PATH,
    confidence_threshold=0.4,
    max_det=20,
    image_size=640,
    modality="thermal"
)


image_files = sorted(os.listdir(IMAGE_DIR))[:5]


for image_name in image_files:

    image_path = os.path.join(IMAGE_DIR, image_name)

    print("\n" + "=" * 60)
    print(f"IMAGE: {image_name}")
    print("=" * 60)

    detections = detector.predict(image_path)

    if len(detections) == 0:
        print("No detections found.")

    else:
        for idx, det in enumerate(detections):
            print(f"\nDetection {idx + 1}")
            print(det)