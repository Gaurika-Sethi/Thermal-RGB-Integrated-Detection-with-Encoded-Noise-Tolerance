
from pathlib import Path

import cv2
import numpy as np


# Project paths
ROOT = Path(__file__).resolve().parents[2]

RGB_VIDEO = ROOT / "data" / "raw" / "rgb_vid.mp4"
THERMAL_VIDEO = ROOT / "data" / "raw" / "thermal_vid.mp4"

OUTPUT_DIR = ROOT / "outputs" / "calibration"
DEBUG_DIR = OUTPUT_DIR / "debug"

# Number of INNER corners: (columns, rows), not number of squares.
CHECKERBOARD_SIZE = (9, 6)

# Examine every Nth synchronized frame.
FRAME_STRIDE = 10

# Use the more robust classic checkerboard detector.
FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
)


def find_corners(frame):
    """Detect checkerboard inner corners in one image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_SIZE,
        FLAGS,
    )

    if not found:
        return None

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        ),
    )

    return corners.reshape(-1, 2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if not RGB_VIDEO.exists():
        raise FileNotFoundError(f"RGB video not found: {RGB_VIDEO}")

    if not THERMAL_VIDEO.exists():
        raise FileNotFoundError(
            f"Thermal video not found: {THERMAL_VIDEO}"
        )

    rgb_cap = cv2.VideoCapture(str(RGB_VIDEO))
    thermal_cap = cv2.VideoCapture(str(THERMAL_VIDEO))

    if not rgb_cap.isOpened() or not thermal_cap.isOpened():
        raise RuntimeError("Could not open one or both videos.")

    rgb_points = []
    thermal_points = []

    frame_index = 0
    paired_detections = 0

    try:
        while True:
            rgb_ok, rgb_frame = rgb_cap.read()
            thermal_ok, thermal_frame = thermal_cap.read()

            if not rgb_ok or not thermal_ok:
                break

            if frame_index % FRAME_STRIDE == 0:
                rgb_corners = find_corners(rgb_frame)
                thermal_corners = find_corners(thermal_frame)

                if (
                    rgb_corners is not None
                    and thermal_corners is not None
                ):
                    if len(rgb_corners) != len(thermal_corners):
                        print(
                            f"Frame {frame_index}: corner count mismatch"
                        )
                    else:
                        # IMPORTANT:
                        # Inspect the debug images to ensure that
                        # corresponding corners have the same ordering.
                        rgb_points.append(rgb_corners)
                        thermal_points.append(thermal_corners)

                        rgb_debug = rgb_frame.copy()
                        thermal_debug = thermal_frame.copy()

                        cv2.drawChessboardCorners(
                            rgb_debug,
                            CHECKERBOARD_SIZE,
                            rgb_corners.astype(np.float32).reshape(-1, 1, 2),
                            True,
                        )

                        cv2.drawChessboardCorners(
                            thermal_debug,
                            CHECKERBOARD_SIZE,
                            thermal_corners.astype(np.float32).reshape(-1, 1, 2),
                            True,
                        )

                        cv2.imwrite(
                            str(DEBUG_DIR / f"rgb_{frame_index:06d}.jpg"),
                            rgb_debug,
                        )

                        cv2.imwrite(
                            str(DEBUG_DIR / f"thermal_{frame_index:06d}.jpg"),
                            thermal_debug,
                        )

                        paired_detections += 1
                        print(
                            f"Frame {frame_index}: "
                            f"paired checkerboard #{paired_detections}"
                        )

            frame_index += 1

    finally:
        rgb_cap.release()
        thermal_cap.release()

    if paired_detections == 0:
        raise RuntimeError(
            "No paired checkerboard detections found. "
            "Check the pattern size, thermal visibility, "
            "and video synchronization."
        )

    np.savez_compressed(
        OUTPUT_DIR / "checkerboard_points.npz",
        rgb_points=np.array(rgb_points, dtype=np.float32),
        thermal_points=np.array(thermal_points, dtype=np.float32),
        checkerboard_size=np.array(CHECKERBOARD_SIZE),
        frame_stride=np.array(FRAME_STRIDE),
    )

    print("\nDAY 15 CORNER DETECTION SUMMARY")
    print(f"Frames examined: {frame_index}")
    print(f"Paired detections: {paired_detections}")
    print(f"Saved points: {OUTPUT_DIR / 'checkerboard_points.npz'}")
    print(f"Debug images: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
