import os
import json
from collections import Counter
from PIL import Image

# =========================
# CONFIG
# =========================

DATASET_ROOT = "/content/drive/MyDrive/thermal_vision_project/datasets/coco_subset"

TRAIN_IMAGES = os.path.join(DATASET_ROOT, "images/train")
VAL_IMAGES = os.path.join(DATASET_ROOT, "images/val")

TRAIN_LABELS = os.path.join(DATASET_ROOT, "labels/train")
VAL_LABELS = os.path.join(DATASET_ROOT, "labels/val")

TRAIN_JSON = os.path.join(
    DATASET_ROOT,
    "annotations/instances_train2017.json"
)

VAL_JSON = os.path.join(
    DATASET_ROOT,
    "annotations/instances_val2017.json"
)

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]


# =========================
# HELPERS
# =========================

def list_images(folder):
    files = []

    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()

        if ext in VALID_EXTENSIONS:
            files.append(f)

    return files


def list_labels(folder):
    return [
        f for f in os.listdir(folder)
        if f.endswith(".txt")
    ]


def verify_images(folder, image_files):
    corrupted = []

    for img_name in image_files:
        path = os.path.join(folder, img_name)

        try:
            img = Image.open(path)
            img.verify()

        except Exception:
            corrupted.append(img_name)

    return corrupted


def verify_json(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        required_keys = ["images", "annotations", "categories"]

        for key in required_keys:
            if key not in data:
                print(f"❌ Missing key: {key}")

        return data

    except Exception as e:
        print(f"❌ JSON ERROR: {e}")
        return None


def verify_person_only(data):
    invalid = []

    for ann in data["annotations"]:
        if ann["category_id"] != 1:
            invalid.append(ann["category_id"])

    return invalid


def verify_label_matching(images, labels):
    image_stems = set(os.path.splitext(f)[0] for f in images)
    label_stems = set(os.path.splitext(f)[0] for f in labels)

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    return missing_labels, missing_images


def count_annotations(data):
    counter = Counter()

    for ann in data["annotations"]:
        counter[ann["image_id"]] += 1

    return counter


# =========================
# MAIN
# =========================

print("\n========================")
print("DATASET VERIFICATION")
print("========================\n")


# -------------------------
# LIST FILES
# -------------------------

train_images = list_images(TRAIN_IMAGES)
val_images = list_images(VAL_IMAGES)

train_labels = list_labels(TRAIN_LABELS)
val_labels = list_labels(VAL_LABELS)

print(f"✅ Train images: {len(train_images)}")
print(f"✅ Val images:   {len(val_images)}")

print(f"✅ Train labels: {len(train_labels)}")
print(f"✅ Val labels:   {len(val_labels)}")


# -------------------------
# VERIFY IMAGES
# -------------------------

print("\nChecking image integrity...")

bad_train = verify_images(TRAIN_IMAGES, train_images)
bad_val = verify_images(VAL_IMAGES, val_images)

print(f"✅ Corrupted train images: {len(bad_train)}")
print(f"✅ Corrupted val images:   {len(bad_val)}")


# -------------------------
# VERIFY JSON
# -------------------------

print("\nChecking annotation JSON...")

train_data = verify_json(TRAIN_JSON)
val_data = verify_json(VAL_JSON)

if train_data and val_data:
    print("✅ JSON files loaded successfully")


# -------------------------
# VERIFY PERSON ONLY
# -------------------------

print("\nChecking category filtering...")

invalid_train = verify_person_only(train_data)
invalid_val = verify_person_only(val_data)

print(f"✅ Invalid train categories: {len(invalid_train)}")
print(f"✅ Invalid val categories:   {len(invalid_val)}")


# -------------------------
# VERIFY LABEL MATCHING
# -------------------------

print("\nChecking image-label consistency...")

missing_train_labels, missing_train_images = verify_label_matching(
    train_images,
    train_labels
)

missing_val_labels, missing_val_images = verify_label_matching(
    val_images,
    val_labels
)

print(f"✅ Missing train labels: {len(missing_train_labels)}")
print(f"✅ Missing val labels:   {len(missing_val_labels)}")

print(f"✅ Labels without train images: {len(missing_train_images)}")
print(f"✅ Labels without val images:   {len(missing_val_images)}")


# -------------------------
# ANNOTATION STATS
# -------------------------

print("\nComputing annotation statistics...")

train_counter = count_annotations(train_data)
val_counter = count_annotations(val_data)

avg_train = sum(train_counter.values()) / len(train_counter)
avg_val = sum(val_counter.values()) / len(val_counter)

print(f"✅ Train annotations: {sum(train_counter.values())}")
print(f"✅ Val annotations:   {sum(val_counter.values())}")

print(f"✅ Avg annotations/image (train): {avg_train:.2f}")
print(f"✅ Avg annotations/image (val):   {avg_val:.2f}")


# -------------------------
# DONE
# -------------------------

print("\n========================")
print("VERIFICATION COMPLETE")
print("========================\n")