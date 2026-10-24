
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bytetracker import BYTETracker


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"

INPUT_CSV = OUTPUT_DIR / "bytetrack_detections.csv"
BASELINE_CSV = OUTPUT_DIR / "tracked_detections.csv"
SUMMARY_CSV = OUTPUT_DIR / "bytetrack_tuning_summary.csv"

FRAME_RATE = 30

EXPERIMENTS = {
    "test_c": {
        "track_thresh": 0.45,
        "track_buffer": 50,
        "match_thresh": 0.80,
    },
    "test_d": {
        "track_thresh": 0.45,
        "track_buffer": 50,
        "match_thresh": 0.85,
    },
    "test_e": {
        "track_thresh": 0.45,
        "track_buffer": 50,
        "match_thresh": 0.90,
    },
}

def run_experiment(detections, name, params):
    """Run one configuration without overwriting the baseline."""
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

    output_path = OUTPUT_DIR / f"tracked_detections_{name}.csv"
    output = pd.DataFrame(
        results,
        columns=[
            "frame", "x1", "y1", "x2", "y2",
            "track_id", "class_id", "confidence",
        ],
    )

    output.to_csv(output_path, index=False)
    print(f"{name}: saved {len(output)} rows to {output_path.name}")
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
        tracks_with_gaps = int(track_stats["has_internal_gap"].sum())
        total_missing_frames = int(
            track_stats["internal_missing_frames"].sum()
        )
        mean_observed = float(track_stats["observed_frames"].mean())
        median_observed = float(track_stats["observed_frames"].median())

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
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Missing baseline output: {BASELINE_CSV}"
        )

    detections = pd.read_csv(INPUT_CSV)
    baseline = pd.read_csv(BASELINE_CSV)

    if detections.empty or baseline.empty:
        raise ValueError("Input detections and baseline must be nonempty.")

    required = {
        "frame", "x1", "y1", "x2", "y2",
        "confidence", "class_id",
    }
    if not required.issubset(detections.columns):
        raise ValueError(
            f"Input is missing columns: {sorted(required - set(detections.columns))}"
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

    baseline_params = {
        "track_thresh": 0.45,
        "track_buffer": 25,
        "match_thresh": 0.80,
    }

    rows = [
        calculate_metrics(
            "baseline_existing",
            baseline,
            baseline_params,
            BASELINE_CSV,
        )
    ]

    for name, params in EXPERIMENTS.items():
        tracked, output_path = run_experiment(
            detections, name, params
        )
        rows.append(
            calculate_metrics(name, tracked, params, output_path)
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nBYTE TRACK PARAMETER COMPARISON")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
