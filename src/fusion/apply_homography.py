
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MATRIX = (
    ROOT / "outputs" / "calibration"
    / "thermal_to_rgb_homography.npy"
)

DEFAULT_INPUTS = [
    ROOT / "outputs" / "thermal_detections.csv",
    ROOT / "outputs" / "thermal_multiple_people.csv",
]

DEFAULT_OUTPUT = (
    ROOT / "outputs" / "thermal_detections_calibrated.csv"
)


def transform_points(points, H):
    """Transform an Nx2 array of points using a homography."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H)
    return transformed.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Transform thermal detections into RGB coordinates."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()

    input_path = args.input

    if input_path is None:
        input_path = next(
            (path for path in DEFAULT_INPUTS if path.exists()),
            None,
        )

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            "Thermal detection CSV not found. Specify it with --input."
        )

    if not args.matrix.exists():
        raise FileNotFoundError(
            f"Homography matrix not found: {args.matrix}"
        )

    H = np.load(args.matrix).astype(np.float64)

    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Homography must be a finite 3x3 matrix.")

    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Homography matrix is singular.")

    df = pd.read_csv(input_path)
    if "frame" not in df.columns:
        raise ValueError("Input CSV must contain a 'frame' column.")

    has_centroids = {"x", "y"}.issubset(df.columns)
    has_boxes = {"x1", "y1", "x2", "y2"}.issubset(df.columns)

    if not has_centroids and not has_boxes:
        raise ValueError(
            "Expected x,y centroid columns or x1,y1,x2,y2 box columns."
        )

    if "confidence" not in df.columns:
        raise ValueError("Input CSV must contain a 'confidence' column.")

    # Work on a copy and preserve original coordinates for audit.
    result = df.copy()

    if has_boxes:
        box_values = df[["x1", "y1", "x2", "y2"]].to_numpy(
            dtype=float
        )
        valid_boxes = np.isfinite(box_values).all(axis=1)

        # Save original box coordinates.
        for col in ("x1", "y1", "x2", "y2"):
            result[f"original_{col}"] = df[col]

        for col in ("x1", "y1", "x2", "y2"):
            result[col] = np.nan

        if valid_boxes.any():
            b = box_values[valid_boxes]

            # Transform all four corners, then enclose them in
            # an axis-aligned box in RGB coordinates.
            corners = np.stack(
                [
                    b[:, [0, 1]],
                    b[:, [2, 1]],
                    b[:, [2, 3]],
                    b[:, [0, 3]],
                ],
                axis=1,
            )

            warped = transform_points(corners.reshape(-1, 2), H)
            warped = warped.reshape(-1, 4, 2)

            result.loc[valid_boxes, "x1"] = warped[:, :, 0].min(axis=1)
            result.loc[valid_boxes, "y1"] = warped[:, :, 1].min(axis=1)
            result.loc[valid_boxes, "x2"] = warped[:, :, 0].max(axis=1)
            result.loc[valid_boxes, "y2"] = warped[:, :, 1].max(axis=1)

    if has_centroids:
        xy = df[["x", "y"]].to_numpy(dtype=float)
        valid_xy = np.isfinite(xy).all(axis=1)

        result["original_x"] = df["x"]
        result["original_y"] = df["y"]
        result["x"] = np.nan
        result["y"] = np.nan

        if valid_xy.any():
            warped_xy = transform_points(xy[valid_xy], H)
            result.loc[valid_xy, "x"] = warped_xy[:, 0]
            result.loc[valid_xy, "y"] = warped_xy[:, 1]

    elif has_boxes:
        # Produce centroids for the existing fusion loader.
        result["x"] = (result["x1"] + result["x2"]) / 2
        result["y"] = (result["y1"] + result["y2"]) / 2

    # Preserve rows without detections; don't fabricate coordinates.
    result["calibration_applied"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    valid = result["x"].notna() & result["y"].notna()

    print("HOMOGRAPHY APPLICATION COMPLETE")
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Matrix: {args.matrix}")
    print(f"Total rows: {len(result)}")
    print(f"Rows with valid centroids: {int(valid.sum())}")
    print(f"Rows without valid centroids: {int((~valid).sum())}")
    print("Original coordinates retained for audit.")


if __name__ == "__main__":
    main()
