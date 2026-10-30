
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RGB_VIDEO = ROOT / "data/raw/RGB_light_vid.mp4"
IR_VIDEO = ROOT / "data/raw/IR_light_vid.mp4"

OUTPUT_DIR = ROOT / "outputs/calibration/day6_light"
FRAME_INDEX = 1875
NUM_POINTS = 6


def extract_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read frame {frame_index} from {video_path}"
        )
    return frame


def collect_points(image, title, count):
    points = []
    display = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append((float(x), float(y)))
            cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                display, str(len(points)), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            cv2.imshow(title, display)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(image.shape[1], 1200), min(image.shape[0], 800))
    cv2.setMouseCallback(title, on_mouse)

    print(f"\n{title}")
    print(f"Click {count} landmarks in order.")
    print("Press R to reset this image's points.")
    print("Press ENTER when all points are selected.")

    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10) and len(points) == count:
            break

        if key in (ord("r"), ord("R")):
            points.clear()
            display = image.copy()
            cv2.imshow(title, display)

        if key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled by user.")

    cv2.destroyWindow(title)
    return np.asarray(points, dtype=np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb = extract_frame(RGB_VIDEO, FRAME_INDEX)
    ir = extract_frame(IR_VIDEO, FRAME_INDEX)

    cv2.imwrite(str(OUTPUT_DIR / "manual_rgb_frame.jpg"), rgb)
    cv2.imwrite(str(OUTPUT_DIR / "manual_ir_frame.jpg"), ir)

    print("RGB image size:", rgb.shape[1], "x", rgb.shape[0])
    print("IR image size:", ir.shape[1], "x", ir.shape[0])

    print("\nSelect landmarks in the RGB image first.")
    rgb_points = collect_points(rgb, "RGB - select landmarks", NUM_POINTS)

    print("\nNow select the SAME landmarks in the IR image.")
    print("Use exactly the same point order as in the RGB image.")
    ir_points = collect_points(ir, "IR - select corresponding landmarks", NUM_POINTS)

    H, inlier_mask = cv2.findHomography(
        ir_points,
        rgb_points,
        method=0,
    )

    if H is None or not np.isfinite(H).all():
        raise RuntimeError("Homography estimation failed.")

    projected = cv2.perspectiveTransform(
        ir_points.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - rgb_points, axis=1)

    print("\nPoint reprojection errors (pixels):")
    for i, error in enumerate(errors, start=1):
        print(f"Point {i}: {error:.3f}")

    print("Median reprojection error:", float(np.median(errors)))
    print("Maximum reprojection error:", float(np.max(errors)))
    print("\nEstimated homography (original IR -> RGB):")
    print(H)

    np.save(OUTPUT_DIR / "thermal_to_rgb_homography_manual_candidate.npy", H)
    np.savetxt(
        OUTPUT_DIR / "manual_rgb_points.csv",
        rgb_points,
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        OUTPUT_DIR / "manual_ir_points.csv",
        ir_points,
        delimiter=",",
        header="x,y",
        comments="",
    )

    # Visualize the transformed IR frame over the RGB frame.
    warped_ir = cv2.warpPerspective(
        ir, H, (rgb.shape[1], rgb.shape[0])
    )
    overlay = cv2.addWeighted(rgb, 0.5, warped_ir, 0.5, 0)

    cv2.imwrite(str(OUTPUT_DIR / "manual_candidate_overlay.jpg"), overlay)

    print("\nSaved candidate matrix, point coordinates, and overlay.")
    print("This is a CANDIDATE; inspect the overlay before accepting it.")


if __name__ == "__main__":
    main()
