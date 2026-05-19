import json
from collections import Counter


PROJECT_ROOT = "/content/drive/MyDrive/thermal_vision_project"

JSON_PATH = (
    f"{PROJECT_ROOT}/inference_outputs/"
    "thermal_metadata_v1/"
    "batch_detection_metadata.json"
)


# =========================
# LOAD METADATA
# =========================

with open(JSON_PATH, "r") as f:
    records = json.load(f)


print("\n" + "=" * 60)
print("METADATA ANALYSIS")
print("=" * 60)


# =========================
# BASIC STATS
# =========================

total_detections = len(records)

print(f"\nTotal detections: {total_detections}")


# =========================
# CONFIDENCE ANALYSIS
# =========================

confidences = [
    r["confidence"]
    for r in records
]

avg_conf = sum(confidences) / len(confidences)

print(f"\nAverage confidence: {avg_conf:.4f}")

low_conf = [
    c for c in confidences
    if c < 0.5
]

print(f"Low confidence detections: {len(low_conf)}")


# =========================
# AREA ANALYSIS
# =========================

areas = [
    r["area"]
    for r in records
]

small_objects = [
    a for a in areas
    if a < 5000
]

large_objects = [
    a for a in areas
    if a > 50000
]

print(f"\nSmall detections: {len(small_objects)}")

print(f"Large detections: {len(large_objects)}")


# =========================
# ASPECT RATIO ANALYSIS
# =========================

weird_aspect = [
    r for r in records
    if r["aspect_ratio"] < 0.1
    or r["aspect_ratio"] > 1.5
]

print(f"\nWeird aspect ratio detections: {len(weird_aspect)}")


# =========================
# IMAGE-LEVEL ANALYSIS
# =========================

image_counter = Counter()

for r in records:
    image_counter[r["image_name"]] += 1


crowded_images = {
    k: v
    for k, v in image_counter.items()
    if v >= 5
}

print(f"\nCrowded images detected: {len(crowded_images)}")


# =========================
# FLAG POTENTIAL FAILURES
# =========================

print("\n" + "=" * 60)
print("POTENTIAL FAILURE CASES")
print("=" * 60)

print("\nLow confidence examples:")

for r in records[:5]:

    if r["confidence"] < 0.5:

        print(
            r["image_name"],
            r["confidence"]
        )


print("\nWeird aspect ratio examples:")

for r in weird_aspect[:5]:

    print(
        r["image_name"],
        r["aspect_ratio"]
    )