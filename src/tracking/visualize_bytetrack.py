
from pathlib import Path
import argparse

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "IR_dark_vid.mp4"


def color_for_id(track_id):
    """Generate a consistent, visible color for each tracking ID."""
    return (
        60 + (track_id * 37) % 196,
        60 + (track_id * 67) % 196,
        60 + (track_id * 97) % 196,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ByteTrack IDs on a video."
    )

    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(
            f"Video not found: {args.video}\n"
            "Pass the correct video using --video."
        )

    if not args.tracks.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {args.tracks}"
        )

    tracks = pd.read_csv(args.tracks)

    required = {
        "frame", "x1", "y1", "x2", "y2",
        "track_id", "confidence"
    }

    missing = required - set(tracks.columns)

    if missing:
        raise ValueError(
            f"Tracking CSV missing columns: {sorted(missing)}"
        )

    if tracks.empty:
        raise ValueError("Tracking CSV is empty.")

    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)

    if tracks["frame"].min() < 0:
        raise ValueError("Tracking CSV contains negative frame indices.")

    by_frame = {
        frame: group
        for frame, group in tracks.groupby("frame", sort=False)
    }

    cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    if tracks["frame"].max() >= video_frames:
        cap.release()
        raise ValueError(
            "Tracking CSV contains frame indices outside the video range. "
            "Check video and tracking CSV alignment."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {args.output}"
        )

    frame_index = 0
    annotated_frames = 0
    frames_with_tracks = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_tracks = by_frame.get(frame_index)

        if frame_tracks is not None and not frame_tracks.empty:
            frames_with_tracks += 1

            for row in frame_tracks.itertuples(index=False):
                x1 = int(round(row.x1))
                y1 = int(round(row.y1))
                x2 = int(round(row.x2))
                y2 = int(round(row.y2))

                track_id = int(row.track_id)
                confidence = float(row.confidence)

                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                color = color_for_id(track_id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                label = f"ID {track_id} | {confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2

                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, scale, thickness
                )

                label_top = max(
                    0, y1 - text_h - baseline - 8
                )

                cv2.rectangle(
                    frame,
                    (x1, label_top),
                    (min(width - 1, x1 + text_w + 8), y1),
                    color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, max(text_h, y1 - baseline - 5)),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            annotated_frames += 1

        object_count = (
            0 if frame_tracks is None else len(frame_tracks)
        )

        cv2.putText(
            frame,
            f"{args.name} | Frame: {frame_index} | "
            f"Tracked objects: {object_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print("\nBYTETRACK VIDEO VISUALIZATION")
    print("=" * 48)
    print(f"Configuration:       {args.name}")
    print(f"Input video:         {args.video}")
    print(f"Tracking CSV:        {args.tracks}")
    print(f"Video dimensions:    {width} x {height}")
    print(f"Video FPS:           {fps:.2f}")
    print(f"Video frames:        {video_frames}")
    print(f"Frames processed:    {frame_index}")
    print(f"Frames with tracks:  {frames_with_tracks}")
    print(f"Annotated frames:    {annotated_frames}")
    print(f"Unique track IDs:    {tracks['track_id'].nunique()}")
    print(f"Output video:        {args.output}")

    print("\nVisualization completed successfully.")


if __name__ == "__main__":
    main()
