from pathlib import Path

import pandas as pd

from src.features.hooks import FeatureExtractor
from src.features.entropy import compute_feature_entropy
from src.reliability.feature_builder import DetectionFeatureBuilder


MODEL_PATH = "data/thermal/thermal_best.pt"

IMAGE_FOLDER = "data/thermal/images"

OUTPUT_CSV = "src/reliability/outputs/features.csv"


# Initialize extractor
extractor = FeatureExtractor(MODEL_PATH)

# Register hook
extractor.register_hook(layer_index=15)

# Initialize feature builder
builder = DetectionFeatureBuilder(
    modality_flag=1  # thermal
)

all_rows = []

image_paths = list(Path(IMAGE_FOLDER).glob("*.*"))

print(f"\nFound {len(image_paths)} images\n")


for image_path in image_paths:

    print("=" * 50)
    print(f"Processing: {image_path.name}")

    # Run inference
    results = extractor.model.predict(
        str(image_path),
        conf=0.25,
        verbose=False
    )

    # Get features
    features = extractor.features

    if features is None:
        print("No features captured")
        continue

    # Compute entropy
    entropy_value = compute_feature_entropy(features)

    print(f"Entropy: {entropy_value:.4f}")

    # Build detection vectors
    rows = builder.build_features(
        results,
        entropy_value
    )

    print(f"Detections: {len(rows)}")

    all_rows.extend(rows)


# Save CSV
df = pd.DataFrame(all_rows)

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved feature vectors:")
print(OUTPUT_CSV)

# Cleanup
extractor.remove_hook()