import os

# =========================
# CONFIG
# =========================

DATASET_ROOT = "/content/drive/MyDrive/thermal_vision_project/datasets/coco_subset"

TRAIN_LABELS = os.path.join(DATASET_ROOT, "labels/train")
VAL_LABELS = os.path.join(DATASET_ROOT, "labels/val")

# =========================
# VALIDATION
# =========================

def validate_label_file(label_path):

    errors = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines):

        parts = line.strip().split()

        # Must contain exactly 5 values
        if len(parts) != 5:
            errors.append(
                f"Line {line_num+1}: Expected 5 values"
            )
            continue

        try:
            class_id = int(parts[0])

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except:
            errors.append(
                f"Line {line_num+1}: Non-numeric values"
            )
            continue

        # Only person class allowed
        if class_id != 0:
            errors.append(
                f"Line {line_num+1}: Invalid class_id {class_id}"
            )

        # Normalization checks
        values = [x, y, w, h]

        for v in values:
            if v < 0 or v > 1:
                errors.append(
                    f"Line {line_num+1}: Value out of range [0,1]"
                )

        # Width/height checks
        if w <= 0 or h <= 0:
            errors.append(
                f"Line {line_num+1}: Invalid width/height"
            )

    return errors


# =========================
# RUN CHECKS
# =========================

def check_folder(folder):

    print(f"\nChecking: {folder}")

    label_files = [
        f for f in os.listdir(folder)
        if f.endswith(".txt")
    ]

    broken_files = 0
    total_boxes = 0

    for file_name in label_files:

        path = os.path.join(folder, file_name)

        errors = validate_label_file(path)

        with open(path, "r") as f:
            total_boxes += len(f.readlines())

        if errors:

            broken_files += 1

            print(f"\n❌ Errors in {file_name}")

            for err in errors[:5]:
                print("   ", err)

    print(f"\n✅ Total label files: {len(label_files)}")
    print(f"✅ Total boxes: {total_boxes}")
    print(f"✅ Broken files: {broken_files}")


# =========================
# MAIN
# =========================

print("\n========================")
print("YOLO LABEL VERIFICATION")
print("========================")

check_folder(TRAIN_LABELS)
check_folder(VAL_LABELS)

print("\n========================")
print("CHECK COMPLETE")
print("========================")