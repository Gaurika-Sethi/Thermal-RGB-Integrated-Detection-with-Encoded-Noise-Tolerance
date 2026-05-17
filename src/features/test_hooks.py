from pathlib import Path

from hooks import FeatureExtractor
from entropy import compute_feature_entropy


MODEL_PATH = "data/thermal/thermal_best.pt"

IMAGE_FOLDER = "data/thermal/images"


extractor = FeatureExtractor(MODEL_PATH)

# Register hook
extractor.register_hook(layer_index=15)

# Collect images
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

    # Retrieve captured features
    features = extractor.features

    if features is None:
        print("No features captured")
        continue

    print("Feature shape:", features.shape)

    # Debug activation range
    print(
        "Activation range:",
        features.min().item(),
        features.max().item()
    )

    # Compute entropy
    entropy_value = compute_feature_entropy(features)

    print(f"Entropy: {entropy_value:.4f}")


# Cleanup
extractor.remove_hook()