from pathlib import Path
import cv2
import pandas as pd

from corruptions import (
    gaussian_noise,
    contrast_reduction,
    salt_pepper,
    structured_patch,
    sensor_dropout
)


INPUT_FOLDER = "data/thermal/images"

OUTPUT_FOLDER = "src/corruption/outputs"

METADATA_PATH = "src/corruption/metadata/corruptions.csv"

# Output folder
Path(OUTPUT_FOLDER).mkdir(
    parents=True,
    exist_ok=True
)

# Metadata folder
Path(METADATA_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

corruptions = {
    "gaussian_noise": gaussian_noise,
    "contrast_reduction": contrast_reduction,
    "salt_pepper": salt_pepper,
    "structured_patch": structured_patch,
    "sensor_dropout": sensor_dropout
}


severity_levels = [0.05, 0.15, 0.25]

rows = []

image_paths = list(Path(INPUT_FOLDER).glob("*.*"))

print(f"\nFound {len(image_paths)} images\n")


for image_path in image_paths:

    image = cv2.imread(str(image_path))

    for corruption_name, corruption_fn in corruptions.items():

        for severity in severity_levels:

            corrupted = corruption_fn(
                image,
                severity
            )

            output_name = (
                f"{image_path.stem}_"
                f"{corruption_name}_"
                f"{severity}.jpg"
            )

            output_path = (
                Path(OUTPUT_FOLDER) / output_name
            )

            cv2.imwrite(
                str(output_path),
                corrupted
            )

            row = {
                "original_image": image_path.name,
                "corrupted_image": output_name,
                "corruption_type": corruption_name,
                "severity": severity
            }

            rows.append(row)

            print(
                f"Saved: {output_name}"
            )


# Save metadata
df = pd.DataFrame(rows)

df.to_csv(METADATA_PATH, index=False)

print("\nSaved metadata:")
print(METADATA_PATH)