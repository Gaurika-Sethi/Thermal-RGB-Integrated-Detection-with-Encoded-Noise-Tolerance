
from pathlib import Path

import cv2
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "data" / "raw" / "rgb_vid.mp4"
CSV_PATH = ROOT / "outputs" / "fused_detections_calibrated.csv"
OUTPUT_PATH = ROOT / "outputs" / "fused_detections_overlay.mp4"

# BGR colors
SOURCE_COLORS = {
    "both": (0, 220, 0),          # green
    "rgb_only": (255, 120, 0),    # blue/orange-like in BGR
    "thermal_only": (0, 165, 255),# orange
    "none": (180, 180, 180),      # grey
}


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Update VIDEO_PATH to your RGB video."
        )

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fusion CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {
        "frame", "fused_x", "fused_y",
        "fused_confidence", "source"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fusion CSV missing columns: {missing}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    grouped = {
        int(frame): rows
        for frame, rows in df.groupby("frame")
    }

    frame_index = 0
    total_drawn = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rows = grouped.get(frame_index)

            if rows is not None:
                for row in rows.itertuples(index=False):
                    x = getattr(row, "fused_x")
                    y = getattr(row, "fused_y")
                    confidence = getattr(row, "fused_confidence")
                    source = str(getattr(row, "source"))

                    if not (
                        np.isfinite(x)
                        and np.isfinite(y)
                    ):
                        continue

                    x, y = int(round(x)), int(round(y))

                    if not (0 <= x < width and 0 <= y < height):
                        continue

                    color = SOURCE_COLORS.get(
                        source, (255, 255, 255)
                    )

                    cv2.circle(
                        frame, (x, y), 6, color, -1
                    )
                    cv2.circle(
                        frame, (x, y), 9, (255, 255, 255), 1
                    )

                    label = f"{source} {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x + 8, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    total_drawn += 1

            cv2.putText(
                frame,
                f"Frame: {frame_index} | "
                f"Fused detections: {len(rows) if rows is not None else 0}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    print("\nFUSION VISUALIZATION COMPLETE")
    print(f"Video frames processed: {frame_index}")
    print(f"Detection markers drawn: {total_drawn}")
    print(f"Output video: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
