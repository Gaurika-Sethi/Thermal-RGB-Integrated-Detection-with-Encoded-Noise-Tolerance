
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = ROOT / "data" / "raw" / "rgb_vid.mp4"
OUTPUT_DIR = ROOT / "outputs"

EXPERIMENTS = {
    "baseline": OUTPUT_DIR / "tracked_detections.csv",
    "test_e": OUTPUT_DIR / "tracked_detections_test_e.csv",
}

COLORS = [
    (50, 220, 50),
    (255, 160, 40),
    (40, 180, 255),
    (220, 80, 220),
    (255, 220, 50),
    (80, 220, 220),
    (180, 100, 255),
    (255, 100, 100),
]


def render_video(name, csv_path):
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

    tracks = pd.read_csv(csv_path)
    required = {
        "frame", "x1", "y1", "x2", "y2",
        "track_id", "confidence",
    }
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing: {sorted(missing)}")

    tracks["frame"] = pd.to_numeric(tracks["frame"], errors="raise").astype(int)
    tracks["track_id"] = pd.to_numeric(
        tracks["track_id"], errors="raise"
    ).astype(int)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = OUTPUT_DIR / f"bytetrack_comparison_{name}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    grouped = {
        int(frame): group
        for frame, group in tracks.groupby("frame")
    }

    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # Tracking CSV frame numbers are zero-based.
            frame_tracks = grouped.get(frame_number)

            if frame_tracks is not None:
                for row in frame_tracks.itertuples(index=False):
                    x1 = int(round(row.x1))
                    y1 = int(round(row.y1))
                    x2 = int(round(row.x2))
                    y2 = int(round(row.y2))
                    track_id = int(row.track_id)
                    confidence = float(row.confidence)

                    color = COLORS[track_id % len(COLORS)]

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )
                    label = f"ID {track_id} | {confidence:.2f}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"{name.upper()} | Frame {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()

    print(f"\n{name.upper()} visualization complete")
    print(f"Video frames: {frame_number}")
    print(f"Source video frames: {frame_count}")
    print(f"Tracked CSV rows: {len(tracks)}")
    print(f"Unique track IDs: {tracks['track_id'].nunique()}")
    print(f"Saved: {output_path}")


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    for name, csv_path in EXPERIMENTS.items():
        render_video(name, csv_path)


if __name__ == "__main__":
    main()
