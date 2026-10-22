
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FUSED = PROJECT_ROOT / "outputs" / "fused_detections_calibrated.csv"
DEFAULT_RGB = PROJECT_ROOT / "outputs" / "rgb_detections.csv"
DEFAULT_THERMAL = PROJECT_ROOT / "outputs" / "thermal_detections_calibrated.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "bytetrack_detections.csv"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MATCH_TOLERANCE = 2.0  # pixels


def find_source_box(frame_detections, center_x, center_y, modality, frame_number):
    """Find the original bounding box corresponding to a fused detection."""
    if modality == "rgb":
        candidates = frame_detections["rgb"]
    else:
        candidates = frame_detections["thermal"]

    if candidates.empty:
        raise ValueError(
            f"Frame {frame_number}: no {modality.upper()} boxes available "
            f"for a fused detection."
        )

    centers = candidates[["x", "y"]].to_numpy(dtype=float)
    distances = np.sqrt(
        (centers[:, 0] - center_x) ** 2
        + (centers[:, 1] - center_y) ** 2
    )

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance > MATCH_TOLERANCE:
        raise ValueError(
            f"Frame {frame_number}: could not match fused {modality.upper()} "
            f"centroid ({center_x:.3f}, {center_y:.3f}) to a source box. "
            f"Nearest distance: {best_distance:.3f}px."
        )

    return candidates.iloc[best_index]


def main():
    parser = argparse.ArgumentParser(
        description="Convert fused detections into ByteTrack-ready boxes."
    )
    parser.add_argument("--fused", type=Path, default=DEFAULT_FUSED)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--thermal", type=Path, default=DEFAULT_THERMAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fused = pd.read_csv(args.fused)
    rgb = pd.read_csv(args.rgb)
    thermal = pd.read_csv(args.thermal)

    required_fused = {
        "frame", "fused_x", "fused_y", "fused_confidence",
        "source", "rgb_x", "rgb_y", "thermal_x", "thermal_y"
    }
    required_source = {"frame", "x", "y", "width", "height", "confidence"}

    if not required_fused.issubset(fused.columns):
        raise ValueError(
            f"Fused CSV is missing columns: "
            f"{sorted(required_fused - set(fused.columns))}"
        )

    for name, data in [("RGB", rgb), ("thermal", thermal)]:
        if not required_source.issubset(data.columns):
            raise ValueError(
                f"{name} CSV is missing columns: "
                f"{sorted(required_source - set(data.columns))}"
            )

    # Index original detections by frame for efficient lookup.
    rgb_by_frame = {
        int(frame): group.reset_index(drop=True)
        for frame, group in rgb.groupby("frame")
    }
    thermal_by_frame = {
        int(frame): group.reset_index(drop=True)
        for frame, group in thermal.groupby("frame")
    }

    boxes = []
    source_counts = {"both": 0, "rgb_only": 0, "thermal_only": 0}

    for row in fused.itertuples(index=False):
        frame = int(row.frame)
        source = str(row.source).strip().lower()

        if source not in source_counts:
            raise ValueError(
                f"Frame {frame}: unexpected source label {row.source!r}."
            )

        cx = float(row.fused_x)
        cy = float(row.fused_y)
        confidence = float(row.fused_confidence)

        if not np.isfinite([cx, cy, confidence]).all():
            raise ValueError(f"Frame {frame}: non-finite fused values.")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"Frame {frame}: confidence {confidence} is outside [0, 1]."
            )

        frame_sources = {
            "rgb": rgb_by_frame.get(frame, pd.DataFrame()),
            "thermal": thermal_by_frame.get(frame, pd.DataFrame()),
        }

        # For matched detections use the RGB box dimensions; for unmatched
        # detections use the dimensions from the available modality.
        if source in ("both", "rgb_only"):
            box = find_source_box(
                frame_sources, float(row.rgb_x), float(row.rgb_y),
                "rgb", frame
            )
        else:
            box = find_source_box(
                frame_sources, float(row.thermal_x), float(row.thermal_y),
                "thermal", frame
            )

        width = float(box["width"])
        height = float(box["height"])

        if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
            raise ValueError(
                f"Frame {frame}: invalid source box size {width} x {height}."
            )

        # Recenter the selected source box around the fused centroid.
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2

        # Clip to the 1920 x 1080 video frame.
        x1 = float(np.clip(x1, 0, FRAME_WIDTH))
        x2 = float(np.clip(x2, 0, FRAME_WIDTH))
        y1 = float(np.clip(y1, 0, FRAME_HEIGHT))
        y2 = float(np.clip(y2, 0, FRAME_HEIGHT))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Frame {frame}: invalid box after clipping.")

        boxes.append({
            "frame": frame,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "class_id": 0,  # person
            "source": source,
        })
        source_counts[source] += 1

    output = pd.DataFrame(
        boxes,
        columns=[
            "frame", "x1", "y1", "x2", "y2",
            "confidence", "class_id", "source"
        ],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    # Reload the saved file to validate the actual artifact.
    check = pd.read_csv(args.output)

    assert len(check) == len(fused), "Output row count does not match fused input."
    assert check["frame"].notna().all(), "Missing frame numbers."
    assert check[["x1", "y1", "x2", "y2", "confidence"]].notna().all().all()
    assert (check["x1"] >= 0).all() and (check["x2"] <= FRAME_WIDTH).all()
    assert (check["y1"] >= 0).all() and (check["y2"] <= FRAME_HEIGHT).all()
    assert (check["x2"] > check["x1"]).all()
    assert (check["y2"] > check["y1"]).all()
    assert check["confidence"].between(0, 1).all()
    assert (check["class_id"] == 0).all()

    print("DAY 24 - BYTETRACK INPUT PREPARATION")
    print("=" * 48)
    print(f"Fused input rows:       {len(fused)}")
    print(f"Output rows:            {len(check)}")
    print(f"Unique frames:          {check['frame'].nunique()}")
    print(f"Frame range:            {check['frame'].min()} - {check['frame'].max()}")
    print(f"Source distribution:    {source_counts}")
    print(f"Output columns:         {list(check.columns)}")
    print(f"Output saved to:        {args.output}")
    print("Validation:             PASSED")


if __name__ == "__main__":
    main()
