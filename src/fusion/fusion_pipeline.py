

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from frame_alignment import load_detection_csv
from fusion import fuse


DISTANCE_THRESHOLD = 150.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def resolve_input(candidates):
    """Return the first existing path from a list of candidates."""

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def get_valid_detections(frame_df):
    """
    Remove no-detection rows and invalid centroid coordinates.

    A valid detection must have finite x, y, and confidence,
    with confidence greater than zero.
    """

    if frame_df.empty:
        return frame_df.copy()

    valid = (
        frame_df["x"].notna()
        & frame_df["y"].notna()
        & frame_df["confidence"].notna()
    )

    valid &= (
        np.isfinite(frame_df["x"])
        & np.isfinite(frame_df["y"])
        & np.isfinite(frame_df["confidence"])
        & (frame_df["confidence"] > 0)
    )

    return frame_df.loc[valid].reset_index(drop=True)


def make_output_row(
    frame_number,
    fused_detection,
    source,
    rgb_det=None,
    thermal_det=None,
    match_distance=np.nan,
):
    """Create one auditable fused-detection CSV row."""

    row = {
        "frame": frame_number,
        "fused_x": np.nan,
        "fused_y": np.nan,
        "fused_confidence": np.nan,
        "source": source,
        "match_distance": match_distance,
        "rgb_x": np.nan,
        "rgb_y": np.nan,
        "rgb_confidence": np.nan,
        "thermal_x": np.nan,
        "thermal_y": np.nan,
        "thermal_confidence": np.nan,
    }

    if fused_detection is not None:
        row["fused_x"] = fused_detection[0]
        row["fused_y"] = fused_detection[1]
        row["fused_confidence"] = fused_detection[2]

    if rgb_det is not None:
        row["rgb_x"] = rgb_det["x"]
        row["rgb_y"] = rgb_det["y"]
        row["rgb_confidence"] = rgb_det["confidence"]

    if thermal_det is not None:
        row["thermal_x"] = thermal_det["x"]
        row["thermal_y"] = thermal_det["y"]
        row["thermal_confidence"] = thermal_det["confidence"]

    return row


def fuse_frame_all_persons(
    frame_number,
    rgb_dets,
    thermal_dets,
    distance_threshold=DISTANCE_THRESHOLD,
):
    """
    Fuse detections for one frame.

    Uses greedy nearest-centroid matching. Each detection
    can be matched at most once.

    Returns one row per fused or unmatched detection.
    If neither modality detects anyone, returns one row
    with blank coordinates and source='none'.
    """

    rgb_dets = get_valid_detections(rgb_dets)
    thermal_dets = get_valid_detections(thermal_dets)

    results = []

    # Both modalities have no valid detections.
    if rgb_dets.empty and thermal_dets.empty:
        return [
            make_output_row(
                frame_number=frame_number,
                fused_detection=None,
                source="none",
            )
        ]

    # RGB is empty: preserve every thermal detection.
    if rgb_dets.empty:
        for _, thermal_row in thermal_dets.iterrows():
            thermal_det = thermal_row.to_dict()

            results.append(
                make_output_row(
                    frame_number,
                    tuple(
                        thermal_det[key]
                        for key in ("x", "y", "confidence")
                    ),
                    "thermal_only",
                    thermal_det=thermal_det,
                )
            )

        return results

    # Thermal is empty: preserve every RGB detection.
    if thermal_dets.empty:
        for _, rgb_row in rgb_dets.iterrows():
            rgb_det = rgb_row.to_dict()

            results.append(
                make_output_row(
                    frame_number,
                    tuple(
                        rgb_det[key]
                        for key in ("x", "y", "confidence")
                    ),
                    "rgb_only",
                    rgb_det=rgb_det,
                )
            )

        return results

    # Calculate all pairwise centroid distances.
    rgb_points = rgb_dets[["x", "y"]].to_numpy(dtype=float)
    thermal_points = thermal_dets[["x", "y"]].to_numpy(dtype=float)

    distances = np.sqrt(
        np.sum(
            (rgb_points[:, None, :] - thermal_points[None, :, :]) ** 2,
            axis=2,
        )
    )

    # Sort candidate pairs by distance, nearest first.
    rgb_indices, thermal_indices = np.indices(distances.shape)

    candidates = sorted(
        (
            (distances[i, j], i, j)
            for i, j in zip(
                rgb_indices.ravel(),
                thermal_indices.ravel(),
            )
        ),
        key=lambda item: item[0],
    )

    matched_rgb = set()
    matched_thermal = set()

    # Greedy one-to-one matching within this frame.
    for distance, i, j in candidates:

        if distance > distance_threshold:
            break

        if i in matched_rgb or j in matched_thermal:
            continue

        rgb_det = rgb_dets.iloc[i].to_dict()
        thermal_det = thermal_dets.iloc[j].to_dict()

        rgb_tuple = (
            rgb_det["x"],
            rgb_det["y"],
            rgb_det["confidence"],
        )

        thermal_tuple = (
            thermal_det["x"],
            thermal_det["y"],
            thermal_det["confidence"],
        )

        fused_detection = fuse(rgb_tuple, thermal_tuple)

        # Do not fabricate a measurement for zero total confidence.
        if fused_detection is None:
            continue

        results.append(
            make_output_row(
                frame_number,
                fused_detection,
                "both",
                rgb_det=rgb_det,
                thermal_det=thermal_det,
                match_distance=float(distance),
            )
        )

        matched_rgb.add(i)
        matched_thermal.add(j)

    # Preserve unmatched RGB detections.
    for i, rgb_row in rgb_dets.iterrows():

        if i in matched_rgb:
            continue

        rgb_det = rgb_row.to_dict()

        results.append(
            make_output_row(
                frame_number,
                (rgb_det["x"], rgb_det["y"], rgb_det["confidence"]),
                "rgb_only",
                rgb_det=rgb_det,
            )
        )

    # Preserve unmatched thermal detections.
    for j, thermal_row in thermal_dets.iterrows():

        if j in matched_thermal:
            continue

        thermal_det = thermal_row.to_dict()

        results.append(
            make_output_row(
                frame_number,
                (
                    thermal_det["x"],
                    thermal_det["y"],
                    thermal_det["confidence"],
                ),
                "thermal_only",
                thermal_det=thermal_det,
            )
        )

    return results


def run_fusion(rgb_csv, thermal_csv, output_csv):
    """Run fusion across the complete aligned frame sequence."""

    rgb_df = load_detection_csv(rgb_csv)
    thermal_df = load_detection_csv(thermal_csv)

    # Frame numbers must be valid and non-missing.
    for name, df in (("RGB", rgb_df), ("Thermal", thermal_df)):
        df["frame"] = pd.to_numeric(df["frame"], errors="raise")

        if df["frame"].isna().any():
            raise ValueError(f"{name} CSV contains missing frame numbers.")

        if not np.isfinite(df["frame"]).all():
            raise ValueError(f"{name} CSV contains non-finite frame numbers.")

        if (df["frame"] < 0).any() or not np.equal(
            df["frame"], np.floor(df["frame"])
        ).all():
            raise ValueError(f"{name} CSV must contain non-negative integer frames.")

        df["frame"] = df["frame"].astype(int)

    # Retain all frames, including frames with no detections.
    rgb_groups = {
        frame: group.drop(columns=["frame"]).reset_index(drop=True)
        for frame, group in rgb_df.groupby("frame", sort=True)
    }

    thermal_groups = {
        frame: group.drop(columns=["frame"]).reset_index(drop=True)
        for frame, group in thermal_df.groupby("frame", sort=True)
    }

    all_frames = sorted(set(rgb_groups) | set(thermal_groups))

    fused_rows = []

    for frame_number in all_frames:

        rgb_frame = rgb_groups.get(
            frame_number,
            pd.DataFrame(columns=[
                "x", "y", "width", "height", "confidence"
            ]),
        )

        thermal_frame = thermal_groups.get(
            frame_number,
            pd.DataFrame(columns=[
                "x", "y", "width", "height", "confidence"
            ]),
        )

        fused_rows.extend(
            fuse_frame_all_persons(
                frame_number,
                rgb_frame,
                thermal_frame,
            )
        )

    output_columns = [
        "frame",
        "fused_x",
        "fused_y",
        "fused_confidence",
        "source",
        "match_distance",
        "rgb_x",
        "rgb_y",
        "rgb_confidence",
        "thermal_x",
        "thermal_y",
        "thermal_confidence",
    ]

    fused_df = pd.DataFrame(fused_rows, columns=output_columns)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(output_csv, index=False)

    # Report fusion and matching statistics.
    source_counts = (
        fused_df["source"]
        .value_counts()
        .reindex(
            ["both", "rgb_only", "thermal_only", "none"],
            fill_value=0,
        )
    )

    valid_rgb = get_valid_detections(rgb_df)
    valid_thermal = get_valid_detections(thermal_df)

    print("\n" + "=" * 60)
    print("DAY 19 - FULL-SEQUENCE FUSION COMPLETE")
    print("=" * 60)

    print(f"RGB input rows          : {len(rgb_df)}")
    print(f"Thermal input rows      : {len(thermal_df)}")
    print(f"Valid RGB detections    : {len(valid_rgb)}")
    print(f"Valid thermal detections: {len(valid_thermal)}")
    print(f"Unique frames processed : {len(all_frames)}")
    print(f"Fused CSV rows          : {len(fused_df)}")

    print("\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source:<14}: {count}")

    print(f"\nDistance threshold      : {DISTANCE_THRESHOLD:.1f} pixels")
    print(f"CSV saved to:\n{output_csv}")

    print("\nFirst 10 fused rows:")
    print(fused_df.head(10).to_string(index=False))

    return fused_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run multi-person RGB-thermal fusion."
    )

    parser.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="Path to the RGB detection CSV.",
    )

    parser.add_argument(
        "--thermal",
        type=Path,
        default=None,
        help="Path to the thermal detection CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "fused_detections.csv",
        help="Output path for the fused CSV.",
    )

    args = parser.parse_args()

    rgb_csv = args.rgb or resolve_input([
        OUTPUT_DIR / "rgb_multiple_people.csv",
        OUTPUT_DIR / "rgb_detections.csv",
    ])

    thermal_csv = args.thermal or resolve_input([
        OUTPUT_DIR / "thermal_multiple_people.csv",
        OUTPUT_DIR / "thermal_detections.csv",
    ])

    if not rgb_csv.exists():
        parser.error(
            f"RGB CSV not found: {rgb_csv}. "
            "Pass its path using --rgb."
        )

    if not thermal_csv.exists():
        parser.error(
            f"Thermal CSV not found: {thermal_csv}. "
            "Pass its path using --thermal."
        )

    run_fusion(rgb_csv, thermal_csv, args.output)
