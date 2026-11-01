
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENES = {
    "dark": {
        "input": PROJECT_ROOT / "outputs" / "IR_dark_detection_calibrated_day8.csv",
        "homography": PROJECT_ROOT / "outputs" / "calibration" / "day6_dark" / "thermal_to_rgb_homography.npy",
        "rgb_video": PROJECT_ROOT / "data" / "raw" / "RGB_dark_vid.mp4",
        "output": PROJECT_ROOT / "outputs" / "IR_dark_detection_warped.csv",
    },
    "light": {
        "input": PROJECT_ROOT / "outputs" / "IR_light_detection_calibrated_day8.csv",
        "homography": PROJECT_ROOT / "outputs" / "calibration" / "day6_light" / "thermal_to_rgb_homography.npy",
        "rgb_video": PROJECT_ROOT / "data" / "raw" / "RGB_light_vid.mp4",
        "output": PROJECT_ROOT / "outputs" / "IR_light_detection_warped.csv",
    },
}

OVERLAY_ROOT = PROJECT_ROOT / "outputs" / "day12_box_overlays"


def transform_points(points, homography):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)


def warp_box(original_x, original_y, width, height, homography):
    half_w = width / 2.0
    half_h = height / 2.0

    corners = [
        [original_x - half_w, original_y - half_h],
        [original_x + half_w, original_y - half_h],
        [original_x + half_w, original_y + half_h],
        [original_x - half_w, original_y + half_h],
    ]

    warped = transform_points(corners, homography)

    if not np.isfinite(warped).all():
        raise ValueError("Homography produced non-finite box corners.")

    x_min = float(warped[:, 0].min())
    x_max = float(warped[:, 0].max())
    y_min = float(warped[:, 1].min())
    y_max = float(warped[:, 1].max())

    warped_width = x_max - x_min
    warped_height = y_max - y_min

    if warped_width <= 0 or warped_height <= 0:
        raise ValueError("Homography produced an invalid bounding box.")

    return warped, warped_width, warped_height


def create_overlays(scene, source_df, homography, video_path):
    valid = source_df[
        source_df["confidence"].notna()
        & (source_df["confidence"] > 0)
        & source_df["original_x"].notna()
        & source_df["original_y"].notna()
        & source_df["width"].notna()
        & source_df["height"].notna()
    ].copy()

    if valid.empty:
        print(f"{scene}: no valid IR detections for overlays.")
        return

    # Choose ten detections spread across the valid frame range.
    frame_values = np.sort(valid["frame"].astype(int).unique())
    selected_positions = np.linspace(
        0, len(frame_values) - 1, min(10, len(frame_values))
    ).round().astype(int)
    selected_frames = list(dict.fromkeys(frame_values[selected_positions]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RGB video: {video_path}")

    output_dir = OVERLAY_ROOT / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for frame_number in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read RGB frame {frame_number}; skipping.")
            continue

        detections = valid[valid["frame"].astype(int) == int(frame_number)]

        for _, row in detections.iterrows():
            warped, _, _ = warp_box(
                float(row["original_x"]),
                float(row["original_y"]),
                float(row["width"]),
                float(row["height"]),
                homography,
            )

            polygon = np.round(warped).astype(np.int32)
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)

            x1 = int(np.floor(warped[:, 0].min()))
            y1 = int(np.floor(warped[:, 1].min()))
            x2 = int(np.ceil(warped[:, 0].max()))
            y2 = int(np.ceil(warped[:, 1].max()))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{scene.upper()} | frame {frame_number}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        output_path = output_dir / f"frame_{int(frame_number):06d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not save overlay: {output_path}")

        saved += 1

    cap.release()
    print(f"{scene}: saved {saved} overlay images to {output_dir}")


def process_scene(scene):
    config = SCENES[scene]

    for key in ("input", "homography"):
        if not config[key].exists():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")

    df = pd.read_csv(config["input"])
    required = {
        "frame", "x", "y", "width", "height",
        "confidence", "original_x", "original_y",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{scene}: missing input columns: {sorted(missing)}")

    homography = np.load(config["homography"]).astype(np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError(f"{scene}: invalid homography matrix.")

    output = df.copy()
    valid_mask = (
        output["confidence"].notna()
        & (output["confidence"] > 0)
        & output["original_x"].notna()
        & output["original_y"].notna()
        & output["width"].notna()
        & output["height"].notna()
        & (output["width"] > 0)
        & (output["height"] > 0)
    )

    ratios_w = []
    ratios_h = []
    warped_widths = []
    warped_heights = []

    for index, row in output.loc[valid_mask].iterrows():
        _, warped_w, warped_h = warp_box(
            float(row["original_x"]),
            float(row["original_y"]),
            float(row["width"]),
            float(row["height"]),
            homography,
        )

        ratios_w.append(warped_w / float(row["width"]))
        ratios_h.append(warped_h / float(row["height"]))
        warped_widths.append(warped_w)
        warped_heights.append(warped_h)

        output.at[index, "width"] = warped_w
        output.at[index, "height"] = warped_h

    if not ratios_w:
        raise ValueError(f"{scene}: no valid IR boxes were warped.")

    config["output"].parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config["output"], index=False)

    # Reload and verify the saved artifact.
    check = pd.read_csv(config["output"])
    assert len(check) == len(df)
    assert check["x"].equals(df["x"])
    assert check["y"].equals(df["y"])
    assert np.isfinite(check.loc[valid_mask, ["width", "height"]]).all().all()
    assert (check.loc[valid_mask, "width"] > 0).all()
    assert (check.loc[valid_mask, "height"] > 0).all()

    print("\n" + "=" * 60)
    print(f"DAY 12 - {scene.upper()} THERMAL BOX WARP")
    print("=" * 60)
    print(f"Input rows:                 {len(df)}")
    print(f"Valid boxes warped:         {len(ratios_w)}")
    print(
        "Width ratio (warped/raw):   "
        f"median={np.median(ratios_w):.4f}, "
        f"mean={np.mean(ratios_w):.4f}"
    )
    print(
        "Height ratio (warped/raw):  "
        f"median={np.median(ratios_h):.4f}, "
        f"mean={np.mean(ratios_h):.4f}"
    )
    print(f"Warped CSV:                 {config['output']}")

    if scene == "dark":
        print("Dark-scene target: width ratio about 1.0; height ratio about 1.4–1.5.")
        print(
            "Ratio target check: "
            f"{'PASS' if 1.35 <= np.median(ratios_h) <= 1.55 and 0.90 <= np.median(ratios_w) <= 1.10 else 'REVIEW'}"
        )

    if config["rgb_video"].exists():
        create_overlays(scene, df, homography, config["rgb_video"])
    else:
        print(f"RGB video not found; overlays skipped: {config['rgb_video']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        choices=["dark", "light", "both"],
        default="both",
    )
    args = parser.parse_args()

    scenes = ["dark", "light"] if args.scene == "both" else [args.scene]
    for scene in scenes:
        process_scene(scene)


if __name__ == "__main__":
    main()
