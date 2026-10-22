
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "outputs" / "trajectory.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "track_lifecycle_report.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Trajectory CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {"frame", "track_id", "x", "y"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Trajectory CSV missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Trajectory CSV contains no rows.")

    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["track_id"] = pd.to_numeric(
        df["track_id"], errors="raise"
    ).astype(int)

    first_video_frame = int(df["frame"].min())
    last_video_frame = int(df["frame"].max())

    report = (
        df.groupby("track_id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            observed_frames=("frame", "nunique"),
            first_x=("x", "first"),
            first_y=("y", "first"),
            last_x=("x", "last"),
            last_y=("y", "last"),
        )
        .reset_index()
    )

    report["duration_frames"] = (
        report["last_frame"] - report["first_frame"] + 1
    )

    report["missing_frames_inside_track"] = (
        report["duration_frames"] - report["observed_frames"]
    )

    report["spawned_after_video_start"] = (
        report["first_frame"] > first_video_frame
    )

    report["ended_before_video_end"] = (
        report["last_frame"] < last_video_frame
    )

    report["starts_at_video_boundary"] = (
        report["first_frame"] == first_video_frame
    )

    report["ends_at_video_boundary"] = (
        report["last_frame"] == last_video_frame
    )

    report = report.sort_values(
        ["first_frame", "track_id"]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV, index=False)

    check = pd.read_csv(OUTPUT_CSV)

    assert len(check) == df["track_id"].nunique()
    assert not check.duplicated(["track_id"]).any()
    assert (check["last_frame"] >= check["first_frame"]).all()

    print("\nDAY 28 - TRACK LIFECYCLE ANALYSIS")
    print("=" * 52)
    print(f"Video frame range:             {first_video_frame} - {last_video_frame}")
    print(f"Unique track IDs:              {df['track_id'].nunique()}")
    print(
        "Tracks first seen after start: "
        f"{int(report['spawned_after_video_start'].sum())}"
    )
    print(
        "Tracks last seen before end:   "
        f"{int(report['ended_before_video_end'].sum())}"
    )
    print(
        "Tracks starting at boundary:   "
        f"{int(report['starts_at_video_boundary'].sum())}"
    )
    print(
        "Tracks ending at boundary:     "
        f"{int(report['ends_at_video_boundary'].sum())}"
    )
    print(
        "Tracks with internal gaps:     "
        f"{int((report['missing_frames_inside_track'] > 0).sum())}"
    )

    print("\nTrack lifecycle summary:")
    print(
        report[
            [
                "track_id",
                "first_frame",
                "last_frame",
                "duration_frames",
                "missing_frames_inside_track",
                "spawned_after_video_start",
                "ended_before_video_end",
            ]
        ].to_string(index=False)
    )

    print(f"\nReport saved to: {OUTPUT_CSV}")
    print("Structural validation: PASSED")
    print(
        "\nNote: first/last observations are not proof of actual entry "
        "or exit. Confirm these events by watching the corresponding video."
    )


if __name__ == "__main__":
    main()
