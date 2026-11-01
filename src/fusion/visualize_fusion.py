
"""Render adaptive-fusion detections over an RGB video and save contact sheets.

Expected adaptive CSV columns:
    frame, fused_x, fused_y, fused_confidence, source
Optional box columns:
    fused_x1, fused_y1, fused_x2, fused_y2

If box columns are present, rectangles are drawn. Otherwise, centroids are drawn.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "data" / "raw" / "RGB_light_vid.mp4"
DEFAULT_ADAPTIVE = ROOT / "outputs" / "fused_detections_adaptive_light.csv"
DEFAULT_OUT = ROOT / "outputs" / "adaptive_visualization"

COLORS = {
    "both": (0, 220, 0),
    "matched": (0, 220, 0),
    "rgb_only": (255, 140, 0),
    "thermal_only": (0, 165, 255),
    "ir_only": (0, 165, 255),
    "unknown": (255, 255, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize adaptive fusion detections without a baseline."
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--adaptive", default=str(DEFAULT_ADAPTIVE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--contact-frames", type=int, default=16)
    parser.add_argument("--max-video-frames", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.adaptive)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Adaptive CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"frame", "fused_x", "fused_y", "fused_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Adaptive CSV is missing columns: {sorted(missing)}")

    if "source" not in df.columns:
        if "fusion_type" in df.columns:
            df["source"] = df["fusion_type"].replace({
                "matched": "both",
                "ir_only": "thermal_only",
            })
        else:
            df["source"] = "unknown"

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df = df.dropna(subset=["frame", "fused_x", "fused_y"])
    df["frame"] = df["frame"].astype(int)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    video_out = output_dir / "adaptive_overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {video_out}")

    by_frame = {
        int(frame): group
        for frame, group in df.groupby("frame", sort=False)
    }

    max_frames = total_frames
    if args.max_video_frames > 0:
        max_frames = min(max_frames, args.max_video_frames)

    contact_candidates = sorted(by_frame.keys())
    if args.contact_frames > 0 and contact_candidates:
        count = min(args.contact_frames, len(contact_candidates))
        indices = np.linspace(
            0, len(contact_candidates) - 1, count, dtype=int
        )
        selected_frames = [contact_candidates[i] for i in indices]
    else:
        selected_frames = []

    contact_images = []
    frame_index = 0

    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        detections = by_frame.get(frame_index)
        if detections is not None:
            for _, det in detections.iterrows():
                source = str(det["source"]).lower()
                color = COLORS.get(source, COLORS["unknown"])
                confidence = float(det["fused_confidence"])

                box_cols = {
                    "fused_x1", "fused_y1", "fused_x2", "fused_y2"
                }
                if box_cols.issubset(df.columns):
                    x1 = int(round(float(det["fused_x1"])))
                    y1 = int(round(float(det["fused_y1"])))
                    x2 = int(round(float(det["fused_x2"])))
                    y2 = int(round(float(det["fused_y2"])))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_x, label_y = x1, max(20, y1 - 7)
                else:
                    cx = int(round(float(det["fused_x"])))
                    cy = int(round(float(det["fused_y"])))
                    cv2.circle(frame, (cx, cy), 6, color, 2)
                    label_x, label_y = cx + 8, cy - 8

                label = f"{source} {confidence:.2f}"
                cv2.putText(
                    frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    cv2.LINE_AA,
                )

        if frame_index in selected_frames:
            contact_images.append((frame_index, frame.copy()))

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    if contact_images:
        thumb_width = 480
        thumbs = []
        for idx, image in contact_images:
            scale = thumb_width / image.shape[1]
            thumb_height = int(image.shape[0] * scale)
            thumb = cv2.resize(image, (thumb_width, thumb_height))
            cv2.putText(
                thumb, f"Frame {idx}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)

        columns = 4
        rows = int(np.ceil(len(thumbs) / columns))
        cell_h = max(img.shape[0] for img in thumbs)
        sheet = np.zeros(
            (rows * cell_h, columns * thumb_width, 3), dtype=np.uint8
        )

        for i, thumb in enumerate(thumbs):
            r, c = divmod(i, columns)
            y = r * cell_h
            sheet[y:y + thumb.shape[0], c * thumb_width:(c + 1) * thumb_width] = thumb

        sheet_path = output_dir / "contact_sheet_01.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"Contact sheet saved: {sheet_path}")
    else:
        print("No detections found in the selected video frame range.")

    print(f"Video frames processed: {frame_index}")
    print(f"Annotated video saved: {video_out}")


if __name__ == "__main__":
    main()
