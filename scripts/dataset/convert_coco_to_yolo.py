import os
import json
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ANNOTATION_FILE = "data/rgb/annotations/instances_train2017.json"

IMAGE_DIR = "data/dev_slice/rgb/images/train"

LABEL_DIR = "data/dev_slice/rgb/labels/train"

PERSON_CATEGORY_ID = 1

# --------------------------------------------------
# CREATE LABEL OUTPUT DIRECTORY
# --------------------------------------------------

os.makedirs(LABEL_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD COCO JSON
# --------------------------------------------------

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# --------------------------------------------------
# CREATE IMAGE ID → FILE NAME MAP
# --------------------------------------------------

image_info = {}

for img in coco["images"]:
    image_info[img["id"]] = {
        "file_name": img["file_name"],
        "width": img["width"],
        "height": img["height"]
    }

# --------------------------------------------------
# PROCESS ANNOTATIONS
# --------------------------------------------------

for ann in tqdm(coco["annotations"]):

    # keep ONLY persons
    if ann["category_id"] != PERSON_CATEGORY_ID:
        continue

    image_id = ann["image_id"]

    if image_id not in image_info:
        continue

    info = image_info[image_id]

    file_name = info["file_name"]

    image_path = os.path.join(IMAGE_DIR, file_name)

    # skip if image is not in dev slice
    if not os.path.exists(image_path):
        continue

    img_width = info["width"]
    img_height = info["height"]

    # COCO bbox format
    x, y, w, h = ann["bbox"]

    # convert to YOLO format
    x_center = x + w / 2
    y_center = y + h / 2

    # normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    # YOLO class index
    class_id = 0

    # label file path
    label_file = os.path.join(
        LABEL_DIR,
        file_name.replace(".jpg", ".txt")
    )

    # append annotation
    with open(label_file, "a") as f:
        f.write(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{w:.6f} "
            f"{h:.6f}\n"
        )

print("\nConversion complete.")