from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Day 20 within-track gap candidate search.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--context", type=int, default=10)
    args = parser.parse_args()

    if args.max_gap < 1 or args.context < 0:
        raise ValueError("--max-gap must be >= 1 and --context must be >= 0.")
    if not args.trajectory.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {args.trajectory}")
    df = pd.read_csv(args.trajectory)
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Trajectory CSV is empty.")
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["track_id"] = pd.to_numeric(df["track_id"], errors="raise").astype(int)
    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError("Duplicate (frame, track_id) rows found.")

    rows = []
    for track_id, group in df.sort_values(["track_id", "frame"]).groupby("track_id"):
        frames = group["frame"].drop_duplicates().tolist()
        for prev_frame, next_frame in zip(frames, frames[1:]):
            gap = int(next_frame - prev_frame - 1)
            if 1 <= gap <= args.max_gap:
                rows.append({
                    "track_id": int(track_id),
                    "last_seen_frame": int(prev_frame),
                    "first_seen_again_frame": int(next_frame),
                    "missing_frames": gap,
                    "inspect_from_frame": max(int(df.frame.min()), int(prev_frame) - args.context),
                    "inspect_to_frame": min(int(df.frame.max()), int(next_frame) + args.context),
                    "same_id_resumed": "CHECK VISUALLY",
                    "confirmed_occlusion": "NOT YET VERIFIED",
                })

    result = pd.DataFrame(rows, columns=[
        "track_id", "last_seen_frame", "first_seen_again_frame",
        "missing_frames", "inspect_from_frame", "inspect_to_frame",
        "same_id_resumed", "confirmed_occlusion"
    ])
    if not result.empty:
        result = result.sort_values(["missing_frames", "track_id"], ascending=[False, True])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"\nDAY 20 — OCCLUSION CANDIDATES: {args.name}")
    print("=" * 58)
    print(f"Trajectory rows:       {len(df)}")
    print(f"Unique tracks:         {df.track_id.nunique()}")
    print(f"Maximum gap checked:   {args.max_gap} frames")
    print(f"Candidate gaps:        {len(result)}")
    print(f"Report saved:          {args.output}")
    if not result.empty:
        print("\nTop 15 candidates:")
        print(result.head(15).to_string(index=False))
    else:
        print("No within-track gaps found in the configured range.")
    print("\nReminder: these are gap candidates, not confirmed occlusions.")

if __name__ == "__main__":
    main()
