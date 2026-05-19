from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import cv2

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

RGB_MODEL_PATH = "training_outputs/rgb_yolov8s_v1-3/weights/best.pt"
THERMAL_MODEL_PATH = "training_outputs/thermal_best.pt"

RGB_IMAGES = "data/paired_eval/visible"
THERMAL_IMAGES = "data/paired_eval/infrared"

OUTPUT_CSV = "outputs/paired_eval/paired_comparison.csv"

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

print("[INFO] Loading models...")

rgb_model = YOLO(RGB_MODEL_PATH)
thermal_model = YOLO(THERMAL_MODEL_PATH)

# ---------------------------------------------------
# Helper Function
# ---------------------------------------------------

def process_image(model, image_path, modality):
    """
    Run inference and extract simple metadata.
    """

    results = model(image_path, verbose=False)

    detections = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:

            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            bbox_area = width * height

            detections.append({
                "image_name": Path(image_path).name,
                "modality": modality,
                "confidence": round(conf, 4),
                "bbox": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)],
                "bbox_area": round(bbox_area, 2),
                "width": round(width, 2),
                "height": round(height, 2)
            })

    return detections

# ---------------------------------------------------
# Run RGB Inference
# ---------------------------------------------------

all_rows = []

print("[INFO] Running RGB inference...")

rgb_images = sorted(Path(RGB_IMAGES).glob("*"))

for img_path in rgb_images:

    rows = process_image(rgb_model, str(img_path), "rgb")

    all_rows.extend(rows)

# ---------------------------------------------------
# Run Thermal Inference
# ---------------------------------------------------

print("[INFO] Running thermal inference...")

thermal_images = sorted(Path(THERMAL_IMAGES).glob("*"))

for img_path in thermal_images:

    rows = process_image(thermal_model, str(img_path), "thermal")

    all_rows.extend(rows)

# ---------------------------------------------------
# Save CSV
# ---------------------------------------------------

print("[INFO] Saving CSV...")

df = pd.DataFrame(all_rows)

Path("outputs/paired_eval").mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)

print(f"[DONE] Saved comparison CSV -> {OUTPUT_CSV}")
print(df.head())