
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "outputs" / "tracked_detections.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "trajectory.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "frame", "x1", "y1", "x2", "y2", "track_id"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Tracking CSV contains no rows.")

    numeric_columns = [
        "frame", "x1", "y1", "x2", "y2", "track_id"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Input contains missing or non-finite values.")

    if not df["frame"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Frame numbers must be integers.")

    if not df["track_id"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Track IDs must be integers.")

    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError(
            "Duplicate rows found for the same frame and track ID."
        )

    # Calculate the center of each tracked bounding box.
    df["x"] = (df["x1"] + df["x2"]) / 2.0
    df["y"] = (df["y1"] + df["y2"]) / 2.0

    # Calculate velocity separately for each track.
    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    grouped = df.groupby("track_id", sort=False)

    previous_x = grouped["x"].shift(1)
    previous_y = grouped["y"].shift(1)
    previous_frame = grouped["frame"].shift(1)

    frame_delta = df["frame"] - previous_frame

    # Velocity is measured in pixels per frame.
    df["vx"] = (df["x"] - previous_x) / frame_delta
    df["vy"] = (df["y"] - previous_y) / frame_delta

    # First observation of each track has no prior velocity.
    df.loc[frame_delta.isna(), ["vx", "vy"]] = 0.0

    # Retain the core trajectory fields, plus useful box/confidence data.
    output_columns = [
        "frame",
        "track_id",
        "x",
        "y",
        "vx",
        "vy",
    ]

    if "confidence" in df.columns:
        output_columns.append("confidence")

    output = df[output_columns].sort_values(
        ["frame", "track_id"]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    # Validate the saved artifact.
    check = pd.read_csv(OUTPUT_CSV)

    assert len(check) == len(df)
    assert not check[["frame", "track_id", "x", "y", "vx", "vy"]].isna().any().any()
    assert np.isfinite(
        check[["x", "y", "vx", "vy"]].to_numpy(dtype=float)
    ).all()
    assert not check.duplicated(["frame", "track_id"]).any()

    print("\nDAY 27 - TRAJECTORY CSV COMPLETE")
    print("=" * 48)
    print(f"Input tracking rows:    {len(df)}")
    print(f"Trajectory rows:        {len(check)}")
    print(f"Unique track IDs:       {check['track_id'].nunique()}")
    print(f"Frames covered:         {check['frame'].nunique()}")
    print(f"Frame range:            {check['frame'].min()} - {check['frame'].max()}")
    print(f"Output columns:         {list(check.columns)}")
    print(f"Saved to:               {OUTPUT_CSV}")
    print("Validation:             PASSED")


if __name__ == "__main__":
    main()
