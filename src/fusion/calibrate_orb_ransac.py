
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RGB_VIDEO = ROOT / "data/raw/RGB_dark_vid.mp4"
IR_VIDEO = ROOT / "data/raw/IR_dark_vid.mp4"

OUTPUT_DIR = ROOT / "outputs/calibration/day6_dark"
MATRIX_OUT = OUTPUT_DIR / "thermal_to_rgb_homography.npy"
REPORT_OUT = OUTPUT_DIR / "orb_ransac_report.txt"
MATCHES_OUT = OUTPUT_DIR / "best_orb_matches.jpg"
OVERLAY_OUT = OUTPUT_DIR / "best_warped_ir_overlay.jpg"

# Try a frame every second, then include the final frame.
SAMPLE_EVERY_SECONDS = 1.0
MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.30
RANSAC_REPROJECTION_THRESHOLD = 5.0


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (RGB_VIDEO, IR_VIDEO):
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

    rgb_cap = cv2.VideoCapture(str(RGB_VIDEO))
    ir_cap = cv2.VideoCapture(str(IR_VIDEO))

    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        raise RuntimeError("Could not open both video files.")

    try:
        rgb_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ir_count = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
        ir_fps = ir_cap.get(cv2.CAP_PROP_FPS)

        rgb_size = (
            int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        ir_size = (
            int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        print("RGB:", rgb_count, "frames,", rgb_fps, "FPS,", rgb_size)
        print("IR: ", ir_count, "frames,", ir_fps, "FPS,", ir_size)

        print("IR will be resized to RGB dimensions for feature matching.")
        if rgb_count == 0 or ir_count == 0 or rgb_fps <= 0 or ir_fps <= 0:
            raise ValueError("Invalid video metadata.")

        # Sample within the common frame range. Frame-index correspondence
        # is assumed for this initial test and must be confirmed visually.
        common_count = min(rgb_count, ir_count)
        sample_step = max(1, round(min(rgb_fps, ir_fps) * SAMPLE_EVERY_SECONDS))
        sample_indices = list(range(0, common_count, sample_step))
        if common_count - 1 not in sample_indices:
            sample_indices.append(common_count - 1)

        orb = cv2.ORB_create(nfeatures=3000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        best = None
        report_lines = [
            "DAY 6 ORB + RANSAC HOMOGRAPHY REPORT",
            f"RGB video: {RGB_VIDEO}",
            f"IR video: {IR_VIDEO}",
            f"RGB frames/FPS/size: {rgb_count}/{rgb_fps}/{rgb_size}",
            f"IR frames/FPS/size: {ir_count}/{ir_fps}/{ir_size}",
            f"Sampled frames: {len(sample_indices)}",
            "",
        ]

        for frame_index in sample_indices:
            rgb_frame = read_frame(rgb_cap, frame_index)
            ir_frame = read_frame(ir_cap, frame_index)

            if rgb_frame is None or ir_frame is None:
                report_lines.append(
                    f"Frame {frame_index}: skipped (read failure)"
                )
                continue

            # Match features in a common image coordinate space.
            ir_for_matching = cv2.resize(
                ir_frame,
                rgb_size,
                interpolation=cv2.INTER_LINEAR,
            )
            
            rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            ir_gray = cv2.cvtColor(ir_for_matching, cv2.COLOR_BGR2GRAY)

            kp_rgb, desc_rgb = orb.detectAndCompute(rgb_gray, None)
            kp_ir, desc_ir = orb.detectAndCompute(ir_gray, None)

            if desc_rgb is None or desc_ir is None:
                report_lines.append(
                    f"Frame {frame_index}: descriptors unavailable"
                )
                continue

            pairs = matcher.knnMatch(desc_ir, desc_rgb, k=2)
            good = [
                first for first, second in pairs
                if first.distance < 0.75 * second.distance
            ]

            if len(good) < MIN_GOOD_MATCHES:
                report_lines.append(
                    f"Frame {frame_index}: good_matches={len(good)}, "
                    "insufficient for estimation"
                )
                continue

            # Estimate IR -> RGB transformation.
            points_ir = np.float32(
                [kp_ir[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            points_rgb = np.float32(
                [kp_rgb[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                points_ir,
                points_rgb,
                cv2.RANSAC,
                RANSAC_REPROJECTION_THRESHOLD,
            )

            if H is None or mask is None or not np.isfinite(H).all():
                report_lines.append(
                    f"Frame {frame_index}: homography estimation failed"
                )
                continue

            inlier_mask = mask.ravel().astype(bool)
            inlier_count = int(inlier_mask.sum())
            inlier_ratio = inlier_count / len(good)

            projected = cv2.perspectiveTransform(points_ir, H)
            errors = np.linalg.norm(
                projected.reshape(-1, 2) - points_rgb.reshape(-1, 2),
                axis=1,
            )
            inlier_errors = errors[inlier_mask]
            median_error = (
                float(np.median(inlier_errors))
                if len(inlier_errors) else float("inf")
            )

            report_lines.append(
                f"Frame {frame_index}: keypoints_ir={len(kp_ir)}, "
                f"keypoints_rgb={len(kp_rgb)}, good_matches={len(good)}, "
                f"inliers={inlier_count}, inlier_ratio={inlier_ratio:.3f}, "
                f"median_inlier_error={median_error:.3f}px"
            )

            score = (inlier_count, inlier_ratio, -median_error)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "frame_index": frame_index,
                    "H": H,
                    "good": good,
                    "mask": mask,
                    "kp_ir": kp_ir,
                    "kp_rgb": kp_rgb,
                    "ir_frame": ir_for_matching,
                    "rgb_frame": rgb_frame,
                    "inliers": inlier_count,
                    "ratio": inlier_ratio,
                    "median_error": median_error,
                }

        accepted = False
        if best is not None:
            accepted = (
                best["inliers"] >= MIN_INLIERS
                and best["ratio"] >= MIN_INLIER_RATIO
                and best["median_error"] <= RANSAC_REPROJECTION_THRESHOLD
            )

            report_lines.extend([
                "",
                f"Best sampled frame: {best['frame_index']}",
                f"Best inliers: {best['inliers']}",
                f"Best inlier ratio: {best['ratio']:.3f}",
                f"Best median inlier reprojection error: "
                f"{best['median_error']:.3f}px",
                f"Basic quality screening: {'PASSED' if accepted else 'FAILED'}",
                "",
                "H (IR -> RGB):",
                np.array2string(best["H"], precision=8),
            ])

            match_img = cv2.drawMatches(
                best["ir_frame"], best["kp_ir"],
                best["rgb_frame"], best["kp_rgb"],
                best["good"], None,
                matchesMask=best["mask"].ravel().tolist(),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imwrite(str(MATCHES_OUT), match_img)

            warped_ir = cv2.warpPerspective(
                best["ir_frame"], best["H"], best["rgb_frame"].shape[1::-1]
            )
            overlay = cv2.addWeighted(best["rgb_frame"], 0.5, warped_ir, 0.5, 0)
            cv2.imwrite(str(OVERLAY_OUT), overlay)

            if accepted:
                np.save(MATRIX_OUT, best["H"])
                report_lines.append(f"Matrix saved: {MATRIX_OUT}")
            else:
                report_lines.append(
                    "Matrix NOT saved: quality screening failed. "
                    "Inspect the matches and use manual point correspondence "
                    "if needed."
                )
        else:
            report_lines.extend([
                "",
                "No usable homography was found.",
                "Matrix NOT saved. Use manual point correspondence if needed.",
            ])

        report_lines.extend([
            "",
            f"Matches visualization: {MATCHES_OUT}",
            f"Warped overlay: {OVERLAY_OUT}",
            "",
            "Important: a low reprojection error does not prove correct "
            "cross-modal alignment. Inspect stable scene landmarks.",
        ])

        REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")
        print("\n".join(report_lines))
        print(f"\nReport saved: {REPORT_OUT}")

    finally:
        rgb_cap.release()
        ir_cap.release()


if __name__ == "__main__":
    main()
