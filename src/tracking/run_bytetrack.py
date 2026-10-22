
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bytetracker import BYTETracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "outputs" / "bytetrack_detections.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "tracked_detections.csv"

TRACK_THRESHOLD = 0.45
TRACK_BUFFER = 25
MATCH_THRESHOLD = 0.8
FRAME_RATE = 30


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"ByteTrack input CSV not found: {INPUT_CSV}"
        )

    detections = pd.read_csv(INPUT_CSV)

    required_columns = {
        "frame", "x1", "y1", "x2", "y2",
        "confidence", "class_id"
    }

    missing = required_columns - set(detections.columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing)}"
        )

    if detections.empty:
        raise ValueError("Input CSV contains no detections.")

    numeric_columns = [
        "frame", "x1", "y1", "x2", "y2",
        "confidence", "class_id"
    ]

    for column in numeric_columns:
        detections[column] = pd.to_numeric(
            detections[column], errors="raise"
        )

    if detections[numeric_columns].isna().any().any():
        raise ValueError("Input contains missing numeric values.")

    if not np.isfinite(
        detections[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Input contains non-finite numeric values.")

    if not detections["frame"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Frame numbers must be integers.")

    if (detections["frame"] < 0).any():
        raise ValueError("Frame numbers cannot be negative.")

    if not detections["confidence"].between(0, 1).all():
        raise ValueError("Confidence values must be between 0 and 1.")

    if not (detections["x2"] > detections["x1"]).all():
        raise ValueError("Every box must have x2 > x1.")

    if not (detections["y2"] > detections["y1"]).all():
        raise ValueError("Every box must have y2 > y1.")

    if not detections["class_id"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Class IDs must be integers.")

    detections["frame"] = detections["frame"].astype(int)
    detections["class_id"] = detections["class_id"].astype(int)

    detections = detections.sort_values(
        ["frame"], kind="stable"
    ).reset_index(drop=True)

    tracker = BYTETracker(
        track_thresh=TRACK_THRESHOLD,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESHOLD,
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
                    ["x1", "y1", "x2", "y2", "confidence", "class_id"]
                ].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )

        # The second argument is required by this installed API.
        tracked = tracker.update(tracker_input, None)

        # ByteTrack returns:
        # [x1, y1, x2, y2, track_id, class_id, confidence]
        for track in tracked:
            x1, y1, x2, y2 = map(float, track[:4])
            track_id = int(track[4])
            class_id = int(track[5])
            confidence = float(track[6])

            results.append({
                "frame": frame_number,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "track_id": track_id,
                "class_id": class_id,
                "confidence": confidence,
            })

        if (frame_number - first_frame + 1) % 50 == 0:
            print(
                f"Processed frame {frame_number} / {last_frame}"
            )

    output_columns = [
        "frame", "x1", "y1", "x2", "y2",
        "track_id", "class_id", "confidence"
    ]

    output = pd.DataFrame(results, columns=output_columns)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    # Reload and validate the saved artifact.
    check = pd.read_csv(OUTPUT_CSV)

    if not check.empty:
        assert check["track_id"].notna().all()
        assert (check["track_id"] > 0).all()
        assert check["frame"].between(first_frame, last_frame).all()
        assert (check["x2"] > check["x1"]).all()
        assert (check["y2"] > check["y1"]).all()
        assert check["confidence"].between(0, 1).all()

    print("\nDAY 25 - BYTETRACK COMPLETE")
    print("=" * 48)
    print(f"Input detections:       {len(detections)}")
    print(f"Frames processed:       {last_frame - first_frame + 1}")
    print(f"Tracked output rows:    {len(check)}")
    print(f"Unique track IDs:       {check['track_id'].nunique()}")
    print(f"Frame range:            {first_frame} - {last_frame}")
    print(f"Output saved to:        {OUTPUT_CSV}")
    print("Validation:             PASSED")


if __name__ == "__main__":
    main()
