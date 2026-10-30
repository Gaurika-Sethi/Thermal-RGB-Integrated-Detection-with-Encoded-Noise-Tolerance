
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "outputs/IR_light_detection.csv"
OUTPUT = ROOT / "outputs/IR_light_detection_calibrated_day8.csv"

RGB_WIDTH = 1920
RGB_HEIGHT = 1080


def main():
    original = pd.read_csv(INPUT)
    calibrated = pd.read_csv(OUTPUT)

    print("DAY 8 - HOMOGRAPHY OUTPUT AUDIT")
    print("=" * 48)

    assert len(original) == len(calibrated), "Row count changed"
    assert original["frame"].equals(calibrated["frame"]), "Frame order changed"

    required = {
        "original_x", "original_y", "x", "y",
        "confidence", "calibration_applied",
    }
    missing = required - set(calibrated.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"

    valid_original = (
        original["x"].notna()
        & original["y"].notna()
        & (original["confidence"] > 0)
    )
    valid_output = (
        calibrated["x"].notna()
        & calibrated["y"].notna()
        & (calibrated["confidence"] > 0)
    )

    print("Input rows:", len(original))
    print("Output rows:", len(calibrated))
    print("Input valid detections:", int(valid_original.sum()))
    print("Output valid detections:", int(valid_output.sum()))

    if not np.array_equal(valid_original.to_numpy(), valid_output.to_numpy()):
        raise AssertionError("Valid detection rows changed during calibration")

    xy = calibrated.loc[valid_output, ["x", "y"]].to_numpy(dtype=float)

    if not np.isfinite(xy).all():
        raise AssertionError("Non-finite calibrated coordinates found")

    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < RGB_WIDTH)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < RGB_HEIGHT)
    )

    print("Coordinates within RGB frame:", int(inside.sum()))
    print("Coordinates outside RGB frame:", int((~inside).sum()))

    print("\nSample calibrated rows:")
    print(
        calibrated.loc[
            valid_output,
            ["frame", "original_x", "original_y", "x", "y", "confidence"],
        ].head(10).to_string(index=False)
    )

    print("\nStructural validation PASSED.")
    print("Coordinate bounds are reported separately; inspect any outliers.")


if __name__ == "__main__":
    main()