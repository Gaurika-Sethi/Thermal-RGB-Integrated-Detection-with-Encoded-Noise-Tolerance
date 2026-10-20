
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RGB = ROOT / "outputs" / "rgb_detections.csv"
DEFAULT_THERMAL = (
    ROOT / "outputs" / "thermal_detections_calibrated.csv"
)
DEFAULT_REPORT = ROOT / "outputs" / "frame_alignment_report.csv"

EXPECTED_FRAMES = 280


def load_detections(path: Path, modality: str) -> pd.DataFrame:
    """Load detections and normalize them to frame, x, y, confidence."""
    if not path.exists():
        raise FileNotFoundError(f"{modality} CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"frame", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{modality} CSV missing columns: {missing}")

    # Use existing centroids when available.
    if {"x", "y"}.issubset(df.columns):
        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["y"], errors="coerce")

    # Otherwise, derive centroids from bounding boxes.
    elif {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        x1 = pd.to_numeric(df["x1"], errors="coerce")
        y1 = pd.to_numeric(df["y1"], errors="coerce")
        x2 = pd.to_numeric(df["x2"], errors="coerce")
        y2 = pd.to_numeric(df["y2"], errors="coerce")

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

    else:
        raise ValueError(
            f"{modality} CSV needs x,y centroids or "
            "x1,y1,x2,y2 bounding boxes."
        )

    result = pd.DataFrame({
        "frame": pd.to_numeric(df["frame"], errors="coerce"),
        "x": x,
        "y": y,
        "confidence": pd.to_numeric(
            df["confidence"], errors="coerce"
        ),
    })

    # Ignore explicit blank/no-detection rows, but report invalid data.
    invalid_frame = (
        result["frame"].isna()
        | (result["frame"] < 0)
        | (result["frame"] % 1 != 0)
    )
    if invalid_frame.any():
        raise ValueError(
            f"{modality}: {int(invalid_frame.sum())} invalid frame values."
        )

    result["frame"] = result["frame"].astype(int)

    valid = (
        np.isfinite(result[["x", "y", "confidence"]]).all(axis=1)
        & (result["confidence"] > 0)
    )

    invalid_detection_count = int((~valid).sum())
    if invalid_detection_count:
        print(
            f"WARNING: {modality}: excluding "
            f"{invalid_detection_count} invalid/blank detections."
        )

    result = result.loc[valid].reset_index(drop=True)

    if result.empty:
        raise ValueError(f"{modality}: no valid detections found.")

    return result


def nearest_distances(rgb_points, thermal_points):
    """Nearest thermal-centroid distance for each RGB centroid."""
    differences = (
        rgb_points[:, None, :] - thermal_points[None, :, :]
    )
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    return distances.min(axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Verify RGB/thermal CSV frame alignment."
    )
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument(
        "--thermal", type=Path, default=DEFAULT_THERMAL
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT
    )
    parser.add_argument(
        "--expected-frames", type=int, default=EXPECTED_FRAMES
    )
    args = parser.parse_args()

    rgb = load_detections(args.rgb, "RGB")
    thermal = load_detections(args.thermal, "Thermal")

    expected = set(range(args.expected_frames))
    rgb_frames = set(rgb["frame"].unique())
    thermal_frames = set(thermal["frame"].unique())

    print("\nINPUT SUMMARY")
    print(f"RGB file: {args.rgb}")
    print(f"RGB valid detections: {len(rgb)}")
    print(f"RGB unique frames: {len(rgb_frames)}")
    print(f"Thermal file: {args.thermal}")
    print(f"Thermal valid detections: {len(thermal)}")
    print(f"Thermal unique frames: {len(thermal_frames)}")

    print("\nFRAME COVERAGE")
    print("RGB missing expected frames:", sorted(expected - rgb_frames))
    print(
        "Thermal missing expected frames:",
        sorted(expected - thermal_frames),
    )
    print("RGB frames outside expected range:", sorted(rgb_frames - expected))
    print(
        "Thermal frames outside expected range:",
        sorted(thermal_frames - expected),
    )

    rgb_counts = rgb.groupby("frame").size().to_dict()
    thermal_counts = thermal.groupby("frame").size().to_dict()

    report_rows = []

    for frame in range(args.expected_frames):
        rgb_frame = rgb[rgb["frame"] == frame]
        thermal_frame = thermal[thermal["frame"] == frame]

        rgb_points = rgb_frame[["x", "y"]].to_numpy(float)
        thermal_points = thermal_frame[["x", "y"]].to_numpy(float)

        distances = np.array([], dtype=float)

        # Diagnostic only: nearest-neighbour distance is not identity matching.
        if len(rgb_points) and len(thermal_points):
            distances = nearest_distances(rgb_points, thermal_points)

        report_rows.append({
            "frame": frame,
            "rgb_count": rgb_counts.get(frame, 0),
            "thermal_count": thermal_counts.get(frame, 0),
            "both_have_detections": (
                len(rgb_points) > 0 and len(thermal_points) > 0
            ),
            "mean_rgb_to_nearest_thermal_px": (
                float(distances.mean()) if len(distances) else np.nan
            ),
            "median_rgb_to_nearest_thermal_px": (
                float(np.median(distances)) if len(distances) else np.nan
            ),
            "max_rgb_to_nearest_thermal_px": (
                float(distances.max()) if len(distances) else np.nan
            ),
        })

    report = pd.DataFrame(report_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    print("\nALIGNMENT SUMMARY")
    print(
        "Frames with detections in both modalities:",
        int(report["both_have_detections"].sum()),
        "/",
        args.expected_frames,
    )
    available = report["mean_rgb_to_nearest_thermal_px"].dropna()
    if len(available):
        print(
            "Median per-frame mean nearest distance (px):",
            round(float(available.median()), 3),
        )
    print(f"Report saved to: {args.report}")

    if rgb_frames == expected and thermal_frames == expected:
        print("FRAME COVERAGE CHECK: PASS")
    else:
        print("FRAME COVERAGE CHECK: REVIEW REQUIRED")


if __name__ == "__main__":
    main()
