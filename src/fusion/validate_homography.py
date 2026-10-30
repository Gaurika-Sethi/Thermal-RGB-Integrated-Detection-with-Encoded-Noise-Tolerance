
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RGB_VIDEO = ROOT / "data/raw/RGB_light_vid.mp4"
IR_VIDEO = ROOT / "data/raw/IR_light_vid.mp4"

HOMOGRAPHY = (
    ROOT / "outputs/calibration/day6_light/"
    "thermal_to_rgb_homography_manual_candidate.npy"
)

OUTPUT_DIR = ROOT / "outputs/calibration/day7_light"

NUM_FRAMES = 10
PANEL_WIDTH = 640


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def add_title(image, title):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def resize_panel(image):
    height, width = image.shape[:2]
    panel_height = round(PANEL_WIDTH * height / width)
    return cv2.resize(image, (PANEL_WIDTH, panel_height))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (RGB_VIDEO, IR_VIDEO, HOMOGRAPHY):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    H = np.load(HOMOGRAPHY)

    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Homography must be a finite 3 x 3 matrix.")

    rgb_cap = cv2.VideoCapture(str(RGB_VIDEO))
    ir_cap = cv2.VideoCapture(str(IR_VIDEO))

    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        raise RuntimeError("Could not open both videos.")

    try:
        rgb_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ir_count = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
        ir_fps = ir_cap.get(cv2.CAP_PROP_FPS)

        rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ir_width = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ir_height = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("DAY 7 - VISUAL HOMOGRAPHY VALIDATION")
        print("=" * 52)
        print(f"RGB: {rgb_count} frames, {rgb_fps} FPS, "
              f"{rgb_width}x{rgb_height}")
        print(f"IR:  {ir_count} frames, {ir_fps} FPS, "
              f"{ir_width}x{ir_height}")

        # Avoid the final frame because the previous video read failed there.
        common_count = min(rgb_count, ir_count)
        if common_count < NUM_FRAMES:
            raise ValueError("Not enough common frames for validation.")

        frame_indices = np.linspace(
            0, common_count - 2, NUM_FRAMES, dtype=int
        ).tolist()

        print("Sampled frame indices:", frame_indices)

        # Valid-pixel mask identifies where the warped IR image has coverage.
        ir_mask = np.full((ir_height, ir_width), 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            ir_mask,
            H,
            (rgb_width, rgb_height),
            flags=cv2.INTER_NEAREST,
        )
        valid_region = warped_mask > 0

        if not valid_region.any():
            raise ValueError(
                "Homography maps no IR pixels into the RGB frame."
            )

        panels = []
        report_rows = []

        for frame_index in frame_indices:
            rgb = read_frame(rgb_cap, frame_index)
            ir = read_frame(ir_cap, frame_index)

            if rgb is None or ir is None:
                print(f"WARNING: skipped unreadable frame {frame_index}")
                report_rows.append({
                    "frame": frame_index,
                    "status": "read_failed",
                })
                continue

            warped_ir = cv2.warpPerspective(
                ir,
                H,
                (rgb_width, rgb_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # Blend only where transformed IR pixels cover the RGB frame.
            overlay = rgb.copy()
            blended = cv2.addWeighted(rgb, 0.5, warped_ir, 0.5, 0)
            overlay[valid_region] = blended[valid_region]

            rgb_panel = add_title(rgb, f"RGB reference | frame {frame_index}")
            ir_panel = add_title(
                warped_ir, f"Warped IR -> RGB coordinates | frame {frame_index}"
            )
            overlay_panel = add_title(
                overlay, f"50/50 overlay | frame {frame_index}"
            )

            # Save full-resolution images for detailed inspection.
            cv2.imwrite(
                str(OUTPUT_DIR / f"frame_{frame_index:04d}_rgb.jpg"),
                rgb_panel,
            )
            cv2.imwrite(
                str(OUTPUT_DIR / f"frame_{frame_index:04d}_warped_ir.jpg"),
                ir_panel,
            )
            cv2.imwrite(
                str(OUTPUT_DIR / f"frame_{frame_index:04d}_overlay.jpg"),
                overlay_panel,
            )

            # A 3-panel row: RGB, warped IR, blended overlay.
            panels.append(np.hstack([
                resize_panel(rgb_panel),
                resize_panel(ir_panel),
                resize_panel(overlay_panel),
            ]))

            report_rows.append({
                "frame": frame_index,
                "rgb_width": rgb_width,
                "rgb_height": rgb_height,
                "ir_width": ir_width,
                "ir_height": ir_height,
                "valid_warped_ir_pixels": int(valid_region.sum()),
                "valid_warped_ir_fraction": float(valid_region.mean()),
                "status": "inspect_visual_alignment",
            })

            print(f"Processed frame {frame_index}")

        if not panels:
            raise RuntimeError("No frames were successfully validated.")

        contact_sheet = np.vstack(panels)
        contact_path = OUTPUT_DIR / "day7_visual_contact_sheet.jpg"
        cv2.imwrite(str(contact_path), contact_sheet)

        report_path = OUTPUT_DIR / "day7_visual_validation_report.csv"
        pd.DataFrame(report_rows).to_csv(report_path, index=False)

        print("\nContact sheet:", contact_path)
        print("Report:", report_path)
        print("Successful frames:", len(panels))
        print("\nNo automatic pass/fail decision was made.")
        print("Inspect fixed landmarks across the 10 frames.")

    finally:
        rgb_cap.release()
        ir_cap.release()


if __name__ == "__main__":
    main()
