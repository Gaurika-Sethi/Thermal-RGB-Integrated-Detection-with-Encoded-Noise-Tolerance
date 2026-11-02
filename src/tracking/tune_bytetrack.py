
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from bytetracker import BYTETracker


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
TUNING_DIR = OUTPUT_DIR / "day18_tuning"

BASELINE_PATHS = {
    "light": OUTPUT_DIR / "tracked_detections_llvip_light_B0.csv",
    "dark": OUTPUT_DIR / "tracked_detections_llvip_dark_B0.csv",
}

FRAME_RATE = 25

EXPERIMENTS = {
    "T1": {
        "track_thresh": 0.35,
        "track_buffer": 25,
        "match_thresh": 0.80,
    },
    "T2": {
        "track_thresh": 0.55,
        "track_buffer": 25,
        "match_thresh": 0.80,
    },
    "T3": {
        "track_thresh": 0.45,
        "track_buffer": 50,
        "match_thresh": 0.80,
    },
    "T4": {
        "track_thresh": 0.45,
        "track_buffer": 25,
        "match_thresh": 0.70,
    },
    "T5": {
        "track_thresh": 0.45,
        "track_buffer": 25,
        "match_thresh": 0.90,
    },
}

BASELINE_PARAMS = {
    "track_thresh": 0.45,
    "track_buffer": 25,
    "match_thresh": 0.80,
}


def run_experiment(detections, scene, name, params):
    """Run one tuning configuration without overwriting existing outputs."""

    tracker = BYTETracker(
        track_thresh=params["track_thresh"],
        track_buffer=params["track_buffer"],
        match_thresh=params["match_thresh"],
        frame_rate=FRAME_RATE,
    )

    results = []

    first_frame = int(detections["frame"].min())
    last_frame = int(detections["frame"].max())

    for frame_number in range(first_frame, last_frame + 1):
        frame_dets = detections[
            detections["frame"] == frame_number
        ]

        if frame_dets.empty:
            tracker_input = torch.empty((0, 6), dtype=torch.float32)
        else:
            tracker_input = torch.tensor(
                frame_dets[
                    [
                        "x1", "y1", "x2", "y2",
                        "confidence", "class_id",
                    ]
                ].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )

        tracked = tracker.update(tracker_input, None)

        for track in tracked:
            results.append({
                "frame": frame_number,
                "x1": float(track[0]),
                "y1": float(track[1]),
                "x2": float(track[2]),
                "y2": float(track[3]),
                "track_id": int(track[4]),
                "class_id": int(track[5]),
                "confidence": float(track[6]),
            })

    output = pd.DataFrame(
        results,
        columns=[
            "frame", "x1", "y1", "x2", "y2",
            "track_id", "class_id", "confidence",
        ],
    )

    output_path = TUNING_DIR / (
        f"tracked_detections_{scene}_{name}.csv"
    )

    output.to_csv(output_path, index=False)

    print(
        f"{scene} {name}: saved {len(output)} rows "
        f"to {output_path.name}"
    )

    return output, output_path


def calculate_metrics(name, tracked, params, output_path):
    """Calculate comparable track continuity statistics."""

    records = []

    for track_id, group in tracked.groupby("track_id"):
        frames = np.sort(group["frame"].unique())
        gaps = np.diff(frames) - 1

        records.append({
            "track_id": int(track_id),
            "observed_frames": len(frames),
            "internal_missing_frames": int(gaps[gaps > 0].sum()),
            "has_internal_gap": bool((gaps > 0).any()),
        })

    track_stats = pd.DataFrame(records)

    if track_stats.empty:
        short_tracks = 0
        tracks_with_gaps = 0
        total_missing_frames = 0
        mean_observed = 0.0
        median_observed = 0.0
    else:
        short_tracks = int(
            (track_stats["observed_frames"] < 5).sum()
        )

        tracks_with_gaps = int(
            track_stats["has_internal_gap"].sum()
        )

        total_missing_frames = int(
            track_stats["internal_missing_frames"].sum()
        )

        mean_observed = float(
            track_stats["observed_frames"].mean()
        )

        median_observed = float(
            track_stats["observed_frames"].median()
        )

    return {
        "experiment": name,
        "track_thresh": params["track_thresh"],
        "track_buffer": params["track_buffer"],
        "match_thresh": params["match_thresh"],
        "tracked_rows": len(tracked),
        "unique_track_ids": int(tracked["track_id"].nunique()),
        "tracks_under_5_frames": short_tracks,
        "tracks_with_internal_gaps": tracks_with_gaps,
        "total_internal_missing_frames": total_missing_frames,
        "mean_observed_frames": round(mean_observed, 2),
        "median_observed_frames": round(median_observed, 2),
        "output_csv": output_path.name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Day 18 ByteTrack parameter tuning"
    )

    parser.add_argument(
        "--scene",
        required=True,
        choices=["light", "dark"],
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the scene's ByteTrack input CSV",
    )

    args = parser.parse_args()

    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    input_path = ROOT / args.input
    baseline_path = BASELINE_PATHS[args.scene]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing detection input: {input_path}"
        )

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Missing Day 17 baseline: {baseline_path}"
        )

    detections = pd.read_csv(input_path)
    baseline = pd.read_csv(baseline_path)

    if detections.empty or baseline.empty:
        raise ValueError(
            "Detection input and baseline must be nonempty."
        )

    required = {
        "frame", "x1", "y1", "x2", "y2",
        "confidence", "class_id",
    }

    missing = required - set(detections.columns)

    if missing:
        raise ValueError(
            f"Input is missing columns: {sorted(missing)}"
        )

    if not {"frame", "track_id"}.issubset(baseline.columns):
        raise ValueError(
            "Baseline must contain frame and track_id columns."
        )

    if not (detections["x2"] > detections["x1"]).all():
        raise ValueError("Input contains boxes with invalid width.")

    if not (detections["y2"] > detections["y1"]).all():
        raise ValueError("Input contains boxes with invalid height.")

    detections["frame"] = detections["frame"].astype(int)
    detections["class_id"] = detections["class_id"].astype(int)

    detections = detections.sort_values(
        ["frame"], kind="stable"
    ).reset_index(drop=True)

    # B0 is loaded from the actual Day 17 result.
    # It is not rerun or overwritten.
    rows = [
        calculate_metrics(
            "B0",
            baseline,
            BASELINE_PARAMS,
            baseline_path,
        )
    ]

    print(f"\nScene: {args.scene}")
    print(f"Input detections: {len(detections)}")
    print(f"Existing B0 rows: {len(baseline)}")
    print("\nRunning five tuning configurations...")

    for name, params in EXPERIMENTS.items():
        tracked, output_path = run_experiment(
            detections,
            args.scene,
            name,
            params,
        )

        rows.append(
            calculate_metrics(
                name,
                tracked,
                params,
                output_path,
            )
        )

    summary = pd.DataFrame(rows)

    summary_path = (
        TUNING_DIR / f"day18_{args.scene}_tuning_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\nBYTE TRACK PARAMETER COMPARISON")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
