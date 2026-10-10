
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_csv(path, required_columns):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    return df


def draw_detections(frame, detections, color, label):
    """Draw bounding boxes and centroids for one modality."""

    for _, row in detections.iterrows():

        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue

        x = float(row["x"])
        y = float(row["y"])
        width = float(row["width"])
        height = float(row["height"])
        confidence = float(row["confidence"])

        x1 = int(round(x - width / 2))
        y1 = int(round(y - height / 2))
        x2 = int(round(x + width / 2))
        y2 = int(round(y + height / 2))

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), color, 2
        )

        cv2.circle(
            frame, (int(round(x)), int(round(y))),
            4, color, -1
        )

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 7, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_fused_positions(frame, fused_rows):
    """
    Draw fused centroids.

    This assumes fused coordinates are expressed in the
    same image coordinate system as the displayed frame.
    """

    height, width = frame.shape[:2]

    for _, row in fused_rows.iterrows():

        if pd.isna(row["fused_x"]) or pd.isna(row["fused_y"]):
            continue

        x = float(row["fused_x"])
        y = float(row["fused_y"])

        if not (0 <= x < width and 0 <= y < height):
            continue

        point = (int(round(x)), int(round(y)))

        cv2.drawMarker(
            frame,
            point,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=3,
        )

        source = str(row["source"])

        cv2.putText(
            frame,
            f"FUSED [{source}]",
            (point[0] + 8, max(point[1] - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )


def resize_for_display(frame, max_width=760, max_height=650):
    """Resize a frame to fit within a fixed display area."""

    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def select_audit_frames(fused_df, total_frames, count=10):

    selected = set()

    matched = fused_df[
        (fused_df["source"] == "both")
        & fused_df["match_distance"].notna()
    ]

    largest_distances = (
        matched.nlargest(5, "match_distance")["frame"]
        .astype(int)
        .tolist()
    )

    selected.update(largest_distances)

    if total_frames > 0:
        evenly_spaced = np.linspace(
            0,
            total_frames - 1,
            num=min(count, total_frames),
            dtype=int,
        )

        selected.update(evenly_spaced.tolist())

    return sorted(
        frame_number
        for frame_number in selected
        if 0 <= frame_number < total_frames
    )


def main(args):
    rgb_csv = load_csv(
        args.rgb_csv,
        ["frame", "x", "y", "width", "height", "confidence"],
    )

    thermal_csv = load_csv(
        args.thermal_csv,
        ["frame", "x", "y", "width", "height", "confidence"],
    )

    fused_csv = load_csv(
        args.fused_csv,
        [
            "frame",
            "fused_x",
            "fused_y",
            "fused_confidence",
            "source",
            "match_distance",
        ],
    )

    rgb_cap = cv2.VideoCapture(str(args.rgb_video))
    thermal_cap = cv2.VideoCapture(str(args.thermal_video))

    if not rgb_cap.isOpened():
        raise RuntimeError(
            f"Could not open RGB video: {args.rgb_video}"
        )

    if not thermal_cap.isOpened():
        rgb_cap.release()
        raise RuntimeError(
            f"Could not open thermal video: {args.thermal_video}"
        )

    try:
        rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        thermal_total = int(
            thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        total_frames = min(rgb_total, thermal_total)

        print("=" * 60)
        print("DAY 20 - VISUAL FUSION VERIFICATION")
        print("=" * 60)
        print(f"RGB video frames     : {rgb_total}")
        print(f"Thermal video frames : {thermal_total}")
        print(f"Common frame range   : 0 to {total_frames - 1}")

        print(
            "\nIMPORTANT: Frame numbers must refer to the same "
            "scene/time in both videos."
        )
        print(
            "Fused markers are drawn in both views only under "
            "the assumption that their image coordinates align."
        )

        audit_frames = select_audit_frames(
            fused_csv,
            total_frames,
        )

        if not audit_frames:
            print("No frames available for verification.")
            return

        print(f"\nAudit frames: {audit_frames}")
        print("\nControls: N/Right = next | P/Left = previous | Q = quit")

        index = 0

        while True:
            frame_number = audit_frames[index]

            rgb_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            thermal_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            rgb_ok, rgb_frame = rgb_cap.read()
            thermal_ok, thermal_frame = thermal_cap.read()

            if not rgb_ok or not thermal_ok:
                print(f"Could not read frame {frame_number}.")
                break

            rgb_dets = rgb_csv[
                rgb_csv["frame"] == frame_number
            ]

            thermal_dets = thermal_csv[
                thermal_csv["frame"] == frame_number
            ]

            fused_rows = fused_csv[
                fused_csv["frame"] == frame_number
            ]

            # Draw modality-specific detections.
            draw_detections(
                rgb_frame,
                rgb_dets,
                (0, 255, 0),
                "RGB",
            )

            draw_detections(
                thermal_frame,
                thermal_dets,
                (255, 255, 0),
                "THERMAL",
            )

            # Overlay fused positions on both views.
            draw_fused_positions(rgb_frame, fused_rows)
            draw_fused_positions(thermal_frame, fused_rows)

            cv2.putText(
                rgb_frame,
                f"RGB | Frame {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                thermal_frame,
                f"THERMAL | Frame {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            # Build a compact per-frame matching summary.
            matched = fused_rows[
                fused_rows["source"] == "both"
            ]

            unmatched_rgb = (
                fused_rows["source"] == "rgb_only"
            ).sum()

            unmatched_thermal = (
                fused_rows["source"] == "thermal_only"
            ).sum()

            max_distance = (
                matched["match_distance"].max()
                if not matched.empty
                else np.nan
            )

            print(
                f"\nFrame {frame_number}: "
                f"matched={len(matched)}, "
                f"RGB-only={unmatched_rgb}, "
                f"thermal-only={unmatched_thermal}, "
                f"max distance={max_distance}"
            )

            # Preserve aspect ratios when displaying.
            rgb_display = resize_for_display(rgb_frame)
            thermal_display = resize_for_display(thermal_frame)
            
            # Give both panels identical dimensions so neither is clipped.
            panel_height = min(
                rgb_display.shape[0],
                thermal_display.shape[0],
            )
            
            rgb_display = cv2.resize(
                rgb_display,
                (
                    int(rgb_display.shape[1] * panel_height / rgb_display.shape[0]),
                    panel_height,
                ),
            )
            
            thermal_display = cv2.resize(
                thermal_display,
                (
                    int(thermal_display.shape[1] * panel_height / thermal_display.shape[0]),
                    panel_height,
                ),
            )
            
            canvas = np.hstack([rgb_display, thermal_display])

            cv2.imshow(
                "Day 20 - RGB | Thermal | Fused",
                canvas,
            )

            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key in (ord("n"), 83):
                index = (index + 1) % len(audit_frames)

            elif key in (ord("p"), 81):
                index = (index - 1) % len(audit_frames)

    finally:
        rgb_cap.release()
        thermal_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visually verify RGB-thermal fusion."
    )

    parser.add_argument(
        "--rgb-video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--thermal-video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--rgb-csv",
        type=Path,
        default=OUTPUT_DIR / "rgb_multiple_people.csv",
    )

    parser.add_argument(
        "--thermal-csv",
        type=Path,
        default=OUTPUT_DIR / "thermal_multiple_people.csv",
    )

    parser.add_argument(
        "--fused-csv",
        type=Path,
        default=OUTPUT_DIR / "fused_detections.csv",
    )

    main(parser.parse_args())
