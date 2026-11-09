from pathlib import Path
import argparse
import pandas as pd
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description="Day 20 track lifecycle analysis.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True,
                        help="Original frame-aligned video; used to establish true video boundaries.")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    if not args.trajectory.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {args.trajectory}")
    df = pd.read_csv(args.trajectory)
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Trajectory CSV is empty.")

    for col in ("frame", "track_id", "x", "y"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not (df[["frame", "track_id", "x", "y"]].notna().all().all()):
        raise ValueError("Trajectory contains missing required values.")
    if ((df["frame"] % 1) != 0).any() or ((df["track_id"] % 1) != 0).any():
        raise ValueError("Frame and track_id values must be integers.")
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError("Duplicate (frame, track_id) rows found.")

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_frame_count <= 0:
        raise ValueError("Could not determine video frame count.")
    video_first, video_last = 0, video_frame_count - 1
    if int(df.frame.min()) < video_first or int(df.frame.max()) > video_last:
        raise ValueError("Trajectory frame indices fall outside the selected video.")
    ordered = df.sort_values(["track_id", "frame"])
    report = ordered.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        observed_frames=("frame", "nunique"),
        first_x=("x", "first"),
        first_y=("y", "first"),
        last_x=("x", "last"),
        last_y=("y", "last"),
    ).reset_index()

    report["duration_frames"] = report["last_frame"] - report["first_frame"] + 1
    report["missing_frames_inside_track"] = report["duration_frames"] - report["observed_frames"]
    report["spawned_after_video_start"] = report["first_frame"] > video_first
    report["ended_before_video_end"] = report["last_frame"] < video_last
    report["starts_at_video_boundary"] = report["first_frame"] == video_first
    report["ends_at_video_boundary"] = report["last_frame"] == video_last
    report["has_internal_gap"] = report["missing_frames_inside_track"] > 0
    report = report.sort_values(["first_frame", "track_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print(f"\nDAY 20 — TRACK LIFECYCLE: {args.name}")
    print("=" * 58)
    print(f"Trajectory rows:              {len(df)}")
    print(f"Video frame range:            {video_first} - {video_last}")
    print(f"Observed trajectory range:    {int(df.frame.min())} - {int(df.frame.max())}")
    print(f"Unique track IDs:             {df.track_id.nunique()}")
    print(f"Tracks first seen after min:  {int(report.spawned_after_video_start.sum())}")
    print(f"Tracks last seen before max:  {int(report.ended_before_video_end.sum())}")
    print(f"Tracks starting at boundary:  {int(report.starts_at_video_boundary.sum())}")
    print(f"Tracks ending at boundary:    {int(report.ends_at_video_boundary.sum())}")
    print(f"Tracks with internal gaps:    {int(report.has_internal_gap.sum())}")
    print(f"Total internal missing frames:{int(report.missing_frames_inside_track.sum()):>7}")
    print(f"Report saved:                 {args.output}")
    print("Structural validation:        PASSED")
    print("Note: first/last observations do not prove physical entry/exit.")

if __name__ == "__main__":
    main()
