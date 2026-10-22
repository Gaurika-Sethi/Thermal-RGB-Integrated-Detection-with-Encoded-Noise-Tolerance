
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "trajectory.csv"
OUTPUT = ROOT / "outputs" / "occlusion_candidates.csv"

MAX_GAP_TO_CHECK = 15  # frames


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {"frame", "track_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(["track_id", "frame"])

    candidates = []

    for track_id, group in df.groupby("track_id"):
        frames = sorted(group["frame"].astype(int).unique())

        for previous_frame, next_frame in zip(frames, frames[1:]):
            gap = next_frame - previous_frame - 1

            if 1 <= gap <= MAX_GAP_TO_CHECK:
                candidates.append({
                    "track_id": int(track_id),
                    "last_seen_frame": previous_frame,
                    "first_seen_again_frame": next_frame,
                    "missing_frames": gap,
                    "inspect_from_frame": max(0, previous_frame - 10),
                    "inspect_to_frame": next_frame + 10,
                    "same_id_resumed": "CHECK VISUALLY",
                    "confirmed_occlusion": "CHECK VIDEO",
                })

    result = pd.DataFrame(candidates)

    if not result.empty:
        result = result.sort_values(
            ["missing_frames", "track_id"],
            ascending=[False, True],
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\nDAY 29 - OCCLUSION CANDIDATE SEARCH")
    print("=" * 48)
    print(f"Trajectory rows:       {len(df)}")
    print(f"Candidate gaps:        {len(result)}")
    print(f"Maximum gap checked:   {MAX_GAP_TO_CHECK} frames")
    print(f"Report saved to:       {OUTPUT}")

    if not result.empty:
        print("\nTop 15 gaps to inspect:")
        print(result.head(15).to_string(index=False))
    else:
        print("No short internal gaps found within the configured range.")


if __name__ == "__main__":
    main()
