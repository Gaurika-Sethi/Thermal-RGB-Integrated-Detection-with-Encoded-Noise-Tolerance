import os
import sys


PROJECT_ROOT = "/content/drive/MyDrive/thermal_vision_project"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from ultralytics import YOLO

from src.reliability.entropy import FeatureExtractor


# =========================
# PATHS
# =========================

MODEL_PATH = f"{PROJECT_ROOT}/training_outputs/thermal_best.pt"

IMAGE_DIR = f"{PROJECT_ROOT}/datasets/LLVIP_YOLO/valid/images"


# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)


# =========================
# FEATURE EXTRACTOR
# =========================

extractor = FeatureExtractor(model)

extractor.register_hooks()


# =========================
# LOAD IMAGE
# =========================

image_files = sorted(os.listdir(IMAGE_DIR))

image_path = os.path.join(
    IMAGE_DIR,
    image_files[0]
)


# =========================
# RUN INFERENCE
# =========================

results = model.predict(
    source=image_path,
    conf=0.4,
    max_det=20,
    imgsz=640,
    verbose=False
)


# =========================
# GET FEATURES
# =========================

features = extractor.get_features()


# =========================
# PRINT INFO
# =========================

print("\n" + "=" * 60)
print("HOOKED FEATURE LAYERS")
print("=" * 60)

for layer_name, tensor in features.items():

    print(f"\n{layer_name}")

    # =====================
    # SINGLE TENSOR
    # =====================

    if hasattr(tensor, "shape"):

        print(f"Shape: {tuple(tensor.shape)}")


    # =====================
    # LIST OF TENSORS
    # =====================

    elif isinstance(tensor, list):

        print(f"List with {len(tensor)} tensors")

        for idx, item in enumerate(tensor):

            if hasattr(item, "shape"):

                print(
                    f"  Tensor {idx}: {tuple(item.shape)}"
                )

# =========================
# CLEANUP
# =========================

extractor.remove_hooks()