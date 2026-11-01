
from pathlib import Path
import json
import numpy as np
import pandas as pd

PARAMS_FILE = Path("C:\\thermal project\\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\\outputs\\calibration\\selected_alpha_beta.json")

SCENES = {
    "light": {
        "fused": Path("C:\\thermal project\\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\\outputs/fused_detections_light.csv"),
        "reliability": Path("C:\\thermal project\\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\\outputs/reliability_scores_light_selected.csv"),
    },
    "dark": {
        "fused": Path("C:\\thermal project\\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\\outputs/fused_detections_dark.csv"),
        "reliability": Path("C:\\thermal project\\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\\outputs/reliability_scores_dark_selected.csv"),
    },
}

COMPATIBILITY_COLUMNS = [
    "frame",
    "fused_x",
    "fused_y",
    "fused_confidence",
    "source",
    "rgb_x",
    "rgb_y",
    "thermal_x",
    "thermal_y",
]

REQUIRED_RELIABILITY_COLUMNS = [
    "frame",
    "rgb_reliability",
    "ir_reliability",
]

all_passed = True


def check(condition, message):
    global all_passed
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        all_passed = False


print("\n" + "=" * 65)
print("DAY 15 - FINAL ADAPTIVE FUSION AUDIT")
print("=" * 65)

# 1. Selected parameters
print("\n1. SELECTED PARAMETERS")

with open(PARAMS_FILE, "r", encoding="utf-8") as f:
    params = json.load(f)

check(params.get("alpha") == 1.0, "alpha = 1.0")
check(params.get("beta") == 0.75, "beta = 0.75")
check(params.get("window_W") == 30, "window W = 30")

print("Loaded parameters:", params)

# 2. Scene outputs
for scene, paths in SCENES.items():
    print(f"\n2. {scene.upper()} SCENE")

    fused = pd.read_csv(paths["fused"])
    reliability = pd.read_csv(paths["reliability"])

    check(not fused.empty, f"{scene}: fused CSV is not empty")

    # Compatibility columns
    check(
        list(fused.columns[:9]) == COMPATIBILITY_COLUMNS,
        f"{scene}: first nine ByteTrack compatibility columns"
    )

    # Required fused fields
    required_fused = [
        "fused_x",
        "fused_y",
        "fused_confidence",
        "source",
        "fusion_type",
    ]

    check(
        all(col in fused.columns for col in required_fused),
        f"{scene}: required fused columns exist"
    )

    # Coordinates and confidence
    for col in ["fused_x", "fused_y", "fused_confidence"]:
        if col in fused.columns:
            values = pd.to_numeric(fused[col], errors="coerce")
            check(
                np.isfinite(values.to_numpy()).all(),
                f"{scene}: {col} has no NaN or infinite values"
            )

    check(
        fused["fused_confidence"].between(0, 1).all(),
        f"{scene}: fused_confidence is within [0, 1]"
    )

    # Fusion labels
    allowed_types = {"matched", "rgb_only", "ir_only"}
    check(
        set(fused["fusion_type"].dropna().unique()).issubset(allowed_types),
        f"{scene}: fusion_type labels are valid"
    )

    print("Fusion distribution:")
    print(fused["fusion_type"].value_counts().to_string())

    # Reliability output
    check(
        all(col in reliability.columns
            for col in REQUIRED_RELIABILITY_COLUMNS),
        f"{scene}: reliability columns exist"
    )

    for col in ["rgb_reliability", "ir_reliability"]:
        values = pd.to_numeric(reliability[col], errors="coerce")
        check(
            np.isfinite(values.to_numpy()).all(),
            f"{scene}: {col} has no NaN or infinite values"
        )
        check(
            values.between(0, 1).all(),
            f"{scene}: {col} is within [0, 1]"
        )

    print(
        "Mean RGB reliability:",
        reliability["rgb_reliability"].mean()
    )
    print(
        "Mean IR reliability:",
        reliability["ir_reliability"].mean()
    )

# 3. Dark reliability direction
dark_rel = pd.read_csv(SCENES["dark"]["reliability"])

dark_rgb_mean = dark_rel["rgb_reliability"].mean()
dark_ir_mean = dark_rel["ir_reliability"].mean()

print("\n3. DARK RELIABILITY THRESHOLDS")

check(dark_rgb_mean < 0.05, "Dark mean RGB reliability < 0.05")
check(dark_ir_mean > 0.3, "Dark mean IR reliability > 0.3")

# 4. Final result
print("\n" + "=" * 65)
if all_passed:
    print("DAY 15 CSV/PARAMETER AUDIT: PASS")
else:
    print("DAY 15 CSV/PARAMETER AUDIT: FAIL")
print("=" * 65)