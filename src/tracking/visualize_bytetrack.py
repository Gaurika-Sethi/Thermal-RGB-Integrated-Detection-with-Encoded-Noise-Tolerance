from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def color_for_id(track_id):
    return (60 + (track_id * 37) % 196,
            60 + (track_id * 67) % 196,
            60 + (track_id * 97) % 196)

def main():
    parser = argparse.ArgumentParser(description="Day 20 ByteTrack ID visualization.")
    parser.add_argument("--video", type=Path, required=True,
                        help="Original video corresponding frame-for-frame to the tracking CSV.")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--homography", type=Path, default=None,
                        help="Supply only when visualizing IR video in RGB coordinates.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.tracks.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {args.tracks}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Output width and height must be positive.")

    H = None
    if args.homography is not None:
        if not args.homography.exists():
            raise FileNotFoundError(f"Homography not found: {args.homography}")
        H = np.load(args.homography)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            raise ValueError("Homography must be a finite 3x3 matrix.")

    tracks = pd.read_csv(args.tracks)
    required = {"frame", "x1", "y1", "x2", "y2", "track_id", "confidence"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Tracking CSV missing columns: {sorted(missing)}")
    if tracks.empty:
        raise ValueError("Tracking CSV is empty.")
    for col in required:
        tracks[col] = pd.to_numeric(tracks[col], errors="raise")
    if not np.isfinite(tracks[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Tracking CSV contains non-finite required values.")
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)
    if tracks.duplicated(["frame", "track_id"]).any():
        raise ValueError("Duplicate (frame, track_id) rows found.")

    by_frame = {int(f): g for f, g in tracks.groupby("frame", sort=False)}
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0
    if tracks.frame.min() < 0 or tracks.frame.max() >= nframes:
        cap.release()
        raise ValueError("Tracking frame indices fall outside the selected video's frame range.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.output.with_suffix(".avi")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (args.width, args.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not initialize output video: {output_path}")

    frame_index = 0
    frames_with_tracks = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if H is not None:
                frame = cv2.warpPerspective(frame, H, (args.width, args.height),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
            elif (frame.shape[1], frame.shape[0]) != (args.width, args.height):
                frame = cv2.resize(frame, (args.width, args.height),
                                   interpolation=cv2.INTER_LINEAR)

            frame_tracks = by_frame.get(frame_index)
            object_count = 0 if frame_tracks is None else len(frame_tracks)
            if object_count:
                frames_with_tracks += 1
                for row in frame_tracks.itertuples(index=False):
                    x1 = max(0, min(int(round(row.x1)), args.width - 1))
                    y1 = max(0, min(int(round(row.y1)), args.height - 1))
                    x2 = max(0, min(int(round(row.x2)), args.width - 1))
                    y2 = max(0, min(int(round(row.y2)), args.height - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    color = color_for_id(int(row.track_id))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {int(row.track_id)} | {float(row.confidence):.2f}"
                    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
                    label_top = max(0, y1 - th - baseline - 8)
                    label_right = min(args.width - 1, x1 + tw + 8)
                    cv2.rectangle(frame, (x1, label_top), (label_right, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 4, max(th, y1 - baseline - 5)),
                                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            cv2.putText(frame, f"{args.name} | Frame: {frame_index} | Tracked objects: {object_count}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    print("\nDAY 20 — BYTETRACK VISUALIZATION")
    print("=" * 52)
    print(f"Configuration:       {args.name}")
    print(f"Input video:         {args.video}")
    print(f"Tracking CSV:        {args.tracks}")
    print(f"Source dimensions:   {source_w} x {source_h}")
    print(f"Output dimensions:   {args.width} x {args.height}")
    print(f"FPS / frames:        {fps:.2f} / {nframes}")
    print(f"Frames processed:    {frame_index}")
    print(f"Frames with tracks:  {frames_with_tracks}")
    print(f"Unique track IDs:    {tracks.track_id.nunique()}")
    print(f"Output video:        {output_path}")
    print("Visualization completed.")

if __name__ == "__main__":
    main()
