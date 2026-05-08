import os
import json
import requests
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------

ANNOTATION_FILE = "data/rgb/annotations/instances_train2017.json"

OUTPUT_DIR = "data/dev_slice/rgb/images/train"

MAX_IMAGES = 500

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD COCO JSON
# -----------------------------

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# -----------------------------
# FIND PERSON CATEGORY ID
# -----------------------------

person_category_id = 1

# -----------------------------
# FIND IMAGE IDS WITH PERSONS
# -----------------------------

person_image_ids = set()

for ann in coco["annotations"]:
    if ann["category_id"] == person_category_id:
        person_image_ids.add(ann["image_id"])

# -----------------------------
# FILTER IMAGE METADATA
# -----------------------------

selected_images = []

for img in coco["images"]:
    if img["id"] in person_image_ids:
        selected_images.append(img)

# limit subset size
selected_images = selected_images[:MAX_IMAGES]

# -----------------------------
# DOWNLOAD IMAGES
# -----------------------------

for img in tqdm(selected_images):

    url = img["coco_url"]
    filename = img["file_name"]

    save_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(save_path):
        continue

    response = requests.get(url)

    with open(save_path, "wb") as f:
        f.write(response.content)

print(f"\nDownloaded {len(selected_images)} images.")