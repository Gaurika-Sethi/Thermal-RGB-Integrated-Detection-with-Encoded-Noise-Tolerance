
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from bytetracker import BYTETracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACK_THRESHOLD = 0.45
TRACK_BUFFER = 25
MATCH_THRESHOLD = 0.80
FRAME_RATE = 25

SHORT_TRACK_THRESHOLD = 5


def calculate_metrics(tracked, input_rows, first_frame, last_frame):
    records = []

    for track_id, group in tracked.groupby("track_id"):
        frames = np.sort(group["frame"].unique())

        observed = len(frames)
        span = int(frames[-1] - frames[0] + 1)
        internal_missing = span - observed

        records.append({
            "track_id": int(track_id),
            "observed_frames": observed,
            "span_frames": span,
            "internal_missing": internal_missing,
        })

    summary = pd.DataFrame(records)

    if summary.empty:
        unique_ids = 0
        short_tracks = 0
        total_internal_missing = 0
        tracks_with_gaps = 0
        mean_track_length = 0.0
    else:
        unique_ids = len(summary)
        short_tracks = int(
            (summary["observed_frames"] < SHORT_TRACK_THRESHOLD).sum()
        )
        total_internal_missing = int(summary["internal_missing"].sum())
        tracks_with_gaps = int(
            (summary["internal_missing"] > 0).sum()
        )
        mean_track_length = float(
            tracked.groupby("track_id")["frame"].nunique().mean()
        )

    return {
        "input_detections": input_rows,
        "tracked_rows": len(tracked),
        "unique_track_ids": unique_ids,
        "short_tracks_under_5_frames": short_tracks,
        "tracks_with_internal_gaps": tracks_with_gaps,
        "total_internal_missing_frames": total_internal_missing,
        "mean_track_length": round(mean_track_length, 2),
        "output_input_ratio": (
            round(len(tracked) / input_rows, 4)
            if input_rows else 0.0
        ),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frames_processed": last_frame - first_frame + 1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the Day 17 ByteTrack baseline."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    detections = pd.read_csv(args.input)

    required_columns = {
        "frame", "x1", "y1", "x2", "y2",
        "confidence", "class_id",
    }

    missing = required_columns - set(detections.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if detections.empty:
        raise ValueError("Input CSV is empty.")

    numeric_columns = list(required_columns)

    for column in numeric_columns:
        detections[column] = pd.to_numeric(
            detections[column], errors="raise"
        )

    values = detections[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite numeric values.")

    if not detections["frame"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Frame numbers must be integers.")

    if (detections["frame"] < 0).any():
        raise ValueError("Negative frame numbers found.")

    if not detections["confidence"].between(0, 1).all():
        raise ValueError("Confidence must be in [0, 1].")

    if not (detections["x2"] > detections["x1"]).all():
        raise ValueError("Invalid bounding-box width.")

    if not (detections["y2"] > detections["y1"]).all():
        raise ValueError("Invalid bounding-box height.")

    detections["frame"] = detections["frame"].astype(int)
    detections["class_id"] = detections["class_id"].astype(int)

    detections = detections.sort_values(
        "frame", kind="stable"
    ).reset_index(drop=True)

    first_frame = int(detections["frame"].min())
    last_frame = int(detections["frame"].max())

    tracker = BYTETracker(
        track_thresh=TRACK_THRESHOLD,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESHOLD,
        frame_rate=FRAME_RATE,
    )

    results = []

    for frame_number in range(first_frame, last_frame + 1):
        frame_dets = detections[
            detections["frame"] == frame_number
        ]

        if frame_dets.empty:
            tracker_input = torch.empty(
                (0, 6), dtype=torch.float32
            )
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

        if (frame_number - first_frame + 1) % 500 == 0:
            print(
                f"Processed frame {frame_number}/{last_frame}"
            )

    output_columns = [
        "frame", "x1", "y1", "x2", "y2",
        "track_id", "class_id", "confidence",
    ]

    output = pd.DataFrame(results, columns=output_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    check = pd.read_csv(args.output)

    if not check.empty:
        assert check["track_id"].notna().all()
        assert (check["track_id"] > 0).all()
        assert check["frame"].between(first_frame, last_frame).all()
        assert (check["x2"] > check["x1"]).all()
        assert (check["y2"] > check["y1"]).all()
        assert check["confidence"].between(0, 1).all()

    metrics = calculate_metrics(
        check, len(detections), first_frame, last_frame
    )

    if args.metrics is None:
        metrics_path = args.output.with_name(
            args.output.stem + "_metrics.csv"
        )
    else:
        metrics_path = args.metrics

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nDAY 17 - BYTETRACK BASELINE")
    print("=" * 52)
    print(f"Input:                  {args.input}")
    print(f"Parameters:             {TRACK_THRESHOLD}, "
          f"{TRACK_BUFFER}, {MATCH_THRESHOLD}, {FRAME_RATE}")
    print(f"Input detections:       {metrics['input_detections']}")
    print(f"Tracked rows:           {metrics['tracked_rows']}")
    print(f"Unique track IDs:       {metrics['unique_track_ids']}")
    print(f"Short tracks (<5):      {metrics['short_tracks_under_5_frames']}")
    print(f"Tracks with gaps:       {metrics['tracks_with_internal_gaps']}")
    print(f"Total internal gaps:    {metrics['total_internal_missing_frames']}")
    print(f"Mean track length:      {metrics['mean_track_length']}")
    print(f"Output/input ratio:     {metrics['output_input_ratio']}")
    print(f"Frame range:            {first_frame} - {last_frame}")
    print(f"Tracked CSV:             {args.output}")
    print(f"Metrics CSV:             {metrics_path}")
    print("Validation:              PASSED")


if __name__ == "__main__":
    main()
