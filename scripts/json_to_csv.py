import json
import glob
import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

JSON_DIR = "inference_outputs/thermal_metadata_v1"
OUTPUT_CSV = "outputs/reliability/reliability_labels.csv"

# =========================================================
# LOAD JSON FILES
# =========================================================

json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

print(f"Found {len(json_files)} JSON files")

rows = []

# =========================================================
# PROCESS FILES
# =========================================================
for json_path in json_files:

    with open(json_path, "r") as f:
        data = json.load(f)

    # -------------------------------------------------
    # EACH ITEM IS ALREADY ONE DETECTION
    # -------------------------------------------------

    for item in data:

        bbox = item.get("bbox", [0, 0, 0, 0])

        x1, y1, x2, y2 = bbox

        width = max(x2 - x1, 1e-6)
        height = max(y2 - y1, 1e-6)

        bbox_area = width * height
        aspect_ratio = width / height

        rows.append({
            "image_path": item.get("image_name", ""),
            "conf": item.get("confidence", 0.0),
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio,

            # placeholder entropy for now
            "entropy": item.get("entropy", 0.0),

            # modality:
            # 0 = rgb
            # 1 = thermal
            "modality": item.get("modality", -1),

            # placeholder reliability
            "reliability": item.get(
                "reliability",
                item.get("confidence", 0.0)
            )
        })

# HANDLE TOP-LEVEL LIST
# -------------------------------------------------

for item in data:

    image_path = item.get("image_path", "")
    modality = item.get("modality", "unknown")

    detections = item.get("detections", [])

    for det in detections:

        # ---------------------------------------------
        # confidence
        # ---------------------------------------------
        conf = det.get("conf", 0.0)

        # ---------------------------------------------
        # bbox
        # expected format:
        # [x1, y1, x2, y2]
        # ---------------------------------------------
        bbox = det.get("bbox", [0, 0, 0, 0])

        x1, y1, x2, y2 = bbox

        width = max(x2 - x1, 1e-6)
        height = max(y2 - y1, 1e-6)

        # ---------------------------------------------
        # derived features
        # ---------------------------------------------
        bbox_area = width * height
        aspect_ratio = width / height

        # ---------------------------------------------
        # entropy
        # ---------------------------------------------
        entropy = det.get("entropy", 0.0)

        # ---------------------------------------------
        # placeholder reliability
        # ---------------------------------------------
        reliability = det.get("reliability", conf)

        # ---------------------------------------------
        # append row
        # ---------------------------------------------
        rows.append({
            "image_path": image_path,
            "conf": conf,
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio,
            "entropy": entropy,
            "modality": modality,
            "reliability": reliability
        })

    for det in detections:

        # ---------------------------------------------
        # confidence
        # ---------------------------------------------
        conf = det.get("conf", 0.0)

        # ---------------------------------------------
        # bbox
        # expected format:
        # [x1, y1, x2, y2]
        # ---------------------------------------------
        bbox = det.get("bbox", [0, 0, 0, 0])

        x1, y1, x2, y2 = bbox

        width = max(x2 - x1, 1e-6)
        height = max(y2 - y1, 1e-6)

        # ---------------------------------------------
        # derived features
        # ---------------------------------------------
        bbox_area = width * height
        aspect_ratio = width / height

        # ---------------------------------------------
        # entropy
        # ---------------------------------------------
        entropy = det.get("entropy", 0.0)

        # ---------------------------------------------
        # placeholder reliability
        # (temporary until RT-DETR labels)
        # ---------------------------------------------
        reliability = det.get("reliability", conf)

        # ---------------------------------------------
        # append row
        # ---------------------------------------------
        rows.append({
            "image_path": image_path,
            "conf": conf,
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio,
            "entropy": entropy,
            "modality": modality,
            "reliability": reliability
        })

# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(rows)

print("\nPreview:")
print(df.head())

# =========================================================
# VALIDATION
# =========================================================

print("\nChecking for NaN values...")
print(df.isnull().sum())

if df.isnull().values.any():
    raise ValueError("NaN values detected!")

print("\nChecking for infinite values...")

numeric_df = df.select_dtypes(include=[np.number])

if not np.isfinite(numeric_df).all().all():
    raise ValueError("Infinite values detected!")

print("Validation passed!")

# =========================================================
# SAVE CSV
# =========================================================

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved CSV to:\n{OUTPUT_CSV}")

print(f"\nTotal rows: {len(df)}")