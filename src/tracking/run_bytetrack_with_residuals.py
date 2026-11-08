from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bytetracker import BYTETracker
from bytetracker import byte_tracker


# Place this script in: src/tracking/
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "day19_residuals"

INPUT_CSV = ROOT / "outputs" / "bytetrack_detections_dark_audit.csv"
FUSED_CSV = ROOT / "outputs" / "fused_detections_dark.csv"

TRACKED_CSV = OUTPUT_DIR / "tracked_detections_dark_B0_residuals.csv"
RESIDUAL_CSV = OUTPUT_DIR / "prediction_residuals_dark_B0.csv"
TRAJECTORY_CSV = OUTPUT_DIR / "trajectory_dark_B0_residuals.csv"
SUMMARY_CSV = OUTPUT_DIR / "residual_summary_dark_B0.csv"

# Day 17/18 Light B0 parameters
TRACK_THRESH = 0.45
TRACK_BUFFER = 25
MATCH_THRESH = 0.80
FRAME_RATE = 25

BOX_MATCH_TOLERANCE = 1e-2
CONF_MATCH_TOLERANCE = 1e-4


def percentile(values, q):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(np.percentile(values, q)) if len(values) else np.nan


def summarize_group(frame, label, gap_label):
    values = pd.to_numeric(frame["prediction_residual_pixels"], errors="coerce").dropna()
    norm = pd.to_numeric(frame["normalized_residual"], errors="coerce").dropna()
    return {
        "gap_group": gap_label,
        "events": int(len(frame)),
        "finite_pixel_residuals": int(len(values)),
        "mean_pixels": float(values.mean()) if len(values) else np.nan,
        "median_pixels": float(values.median()) if len(values) else np.nan,
        "p90_pixels": percentile(values, 90),
        "p95_pixels": percentile(values, 95),
        "finite_normalized_residuals": int(len(norm)),
        "mean_normalized": float(norm.mean()) if len(norm) else np.nan,
        "median_normalized": float(norm.median()) if len(norm) else np.nan,
        "p90_normalized": percentile(norm, 90),
        "p95_normalized": percentile(norm, 95),
    }


def main():
    for path in (INPUT_CSV, FUSED_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    detections = pd.read_csv(INPUT_CSV)
    fused = pd.read_csv(FUSED_CSV)

    detection_columns = {"frame", "x1", "y1", "x2", "y2", "confidence", "class_id"}
    fused_columns = {"frame", "fused_x", "fused_y"}
    if not detection_columns.issubset(detections.columns):
        raise ValueError(f"Input CSV missing columns: {sorted(detection_columns - set(detections.columns))}")
    if not fused_columns.issubset(fused.columns):
        raise ValueError(f"Fusion CSV missing columns: {sorted(fused_columns - set(fused.columns))}")
    if detections.empty or fused.empty:
        raise ValueError("Input and fusion CSVs must be nonempty.")

    numeric_detection_columns = ["frame", "x1", "y1", "x2", "y2", "confidence", "class_id"]
    for col in numeric_detection_columns:
        detections[col] = pd.to_numeric(detections[col], errors="raise")
    for col in ["frame", "fused_x", "fused_y"]:
        fused[col] = pd.to_numeric(fused[col], errors="raise")

    if not np.isfinite(detections[numeric_detection_columns].to_numpy(float)).all():
        raise ValueError("Input detections contain non-finite values.")
    if not np.isfinite(fused[["frame", "fused_x", "fused_y"]].to_numpy(float)).all():
        raise ValueError("Fusion CSV contains non-finite frame/centroid values.")
    if not detections["frame"].map(lambda x: float(x).is_integer()).all():
        raise ValueError("Input frame numbers must be integers.")
    if not fused["frame"].map(lambda x: float(x).is_integer()).all():
        raise ValueError("Fusion frame numbers must be integers.")

    detections["frame"] = detections["frame"].astype(int)
    detections["class_id"] = detections["class_id"].astype(int)
    fused["frame"] = fused["frame"].astype(int)

    # The preparation pipeline preserves row order within each frame.
    if not detections.groupby("frame").size().equals(fused.groupby("frame").size()):
        raise ValueError("Per-frame input/fusion row counts differ; refusing unsafe row matching.")
    if detections["frame"].drop_duplicates().tolist() != fused["frame"].drop_duplicates().tolist():
        raise ValueError("Input/fusion frame order differs.")

    detections = detections.reset_index(drop=True)
    fused = fused.reset_index(drop=True)
    detections["_input_detection_row_id"] = np.arange(len(detections))
    fused["_fused_row_id"] = np.arange(len(fused))

    events, results = [], []
    current_frame = None
    current_frame_dets = None
    current_frame_fused = None
    used_detection_positions = set()
    last_match_frame = {}

    strack = byte_tracker.STrack
    original_multi_predict = strack.multi_predict
    original_update = strack.update
    original_re_activate = strack.re_activate

    def instrumented_multi_predict(stracks):
        result = original_multi_predict(stracks)
        # Installed ByteTrack Kalman state is XYAH: [center_x, center_y, aspect_ratio, height, ...].
        for track in stracks:
            mean = getattr(track, "mean", None)
            if mean is not None and len(mean) >= 4:
                track._day19_predicted_center = np.asarray(mean[:2], dtype=float).copy()
                track._day19_predicted_height = float(mean[3])
                track._day19_prediction_frame = current_frame
        return result

    def find_input_detection_position(new_track):
        if current_frame_dets is None or current_frame_fused is None:
            raise RuntimeError("Current-frame rows were not initialized.")
        new_xywh = np.asarray(new_track._tlwh, dtype=float)
        if new_xywh.shape != (4,) or not np.isfinite(new_xywh).all():
            raise RuntimeError("Matched detection has invalid internal box values.")
        frame_dets = current_frame_dets
        input_xywh = np.column_stack([
            (frame_dets["x1"] + frame_dets["x2"]).to_numpy(float) / 2.0,
            (frame_dets["y1"] + frame_dets["y2"]).to_numpy(float) / 2.0,
            (frame_dets["x2"] - frame_dets["x1"]).to_numpy(float),
            (frame_dets["y2"] - frame_dets["y1"]).to_numpy(float),
        ])
        box_errors = np.max(np.abs(input_xywh - new_xywh[None, :]), axis=1)
        score_errors = np.abs(frame_dets["confidence"].to_numpy(float) - float(new_track.score))
        class_matches = frame_dets["class_id"].to_numpy(int) == int(new_track.cls)
        available = np.ones(len(frame_dets), dtype=bool)
        if used_detection_positions:
            available[list(used_detection_positions)] = False
        candidates = np.flatnonzero(
            available & class_matches & (box_errors <= BOX_MATCH_TOLERANCE)
            & (score_errors <= CONF_MATCH_TOLERANCE)
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Frame {current_frame}: expected exactly one source detection match, got {len(candidates)}; "
                f"best box error={box_errors.min():.6f}, score error={score_errors.min():.6f}."
            )
        pos = int(candidates[0])
        used_detection_positions.add(pos)
        return pos

    def record_match(track, new_track, operation):
        predicted = getattr(track, "_day19_predicted_center", None)
        prediction_frame = getattr(track, "_day19_prediction_frame", None)
        predicted_height = getattr(track, "_day19_predicted_height", np.nan)
        if prediction_frame != current_frame or predicted is None:
            return
        if not np.isfinite(predicted).all() or not np.isfinite(predicted_height) or predicted_height <= 0:
            return

        pos = find_input_detection_position(new_track)
        input_row = current_frame_dets.iloc[pos]
        fused_row = current_frame_fused.iloc[pos]
        det_x = float((input_row.x1 + input_row.x2) / 2.0)
        det_y = float((input_row.y1 + input_row.y2) / 2.0)
        fused_x, fused_y = float(fused_row.fused_x), float(fused_row.fused_y)
        residual = float(np.hypot(fused_x - predicted[0], fused_y - predicted[1]))
        previous_match = last_match_frame.get(int(track.track_id))
        gap_frames = np.nan if previous_match is None else int(current_frame - previous_match - 1)

        events.append({
            "frame": int(current_frame),
            "track_id": int(track.track_id),
            "input_detection_row_id": int(input_row._input_detection_row_id),
            "fused_row_id": int(fused_row._fused_row_id),
            "predicted_x": float(predicted[0]),
            "predicted_y": float(predicted[1]),
            "predicted_height": float(predicted_height),
            "detection_center_x": det_x,
            "detection_center_y": det_y,
            "fused_x": fused_x,
            "fused_y": fused_y,
            "detection_to_fused_distance": float(np.hypot(det_x - fused_x, det_y - fused_y)),
            "prediction_residual_pixels": residual,
            "normalized_residual": residual / float(predicted_height),
            "gap_frames_since_last_match": gap_frames,
            "gap_group": "first_observed_match" if pd.isna(gap_frames) else ("consecutive_k0" if gap_frames == 0 else "after_gap_k_positive"),
            "association": operation,
        })
        last_match_frame[int(track.track_id)] = int(current_frame)

    def instrumented_update(track, new_track, frame_id):
        record_match(track, new_track, "update")
        return original_update(track, new_track, frame_id)

    def instrumented_re_activate(track, new_track, frame_id, new_id=False):
        record_match(track, new_track, "re_activate")
        return original_re_activate(track, new_track, frame_id, new_id=new_id)

    tracker = BYTETracker(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        frame_rate=FRAME_RATE,
    )

    try:
        strack.multi_predict = staticmethod(instrumented_multi_predict)
        strack.update = instrumented_update
        strack.re_activate = instrumented_re_activate
        first_frame, last_frame = int(detections.frame.min()), int(detections.frame.max())

        for frame_number in range(first_frame, last_frame + 1):
            current_frame = frame_number
            current_frame_dets = detections[detections.frame == frame_number].reset_index(drop=True)
            current_frame_fused = fused[fused.frame == frame_number].reset_index(drop=True)
            used_detection_positions = set()
            if len(current_frame_dets) != len(current_frame_fused):
                raise RuntimeError(f"Frame {frame_number}: input/fusion row counts differ.")
            if current_frame_dets.empty:
                tracker_input = torch.empty((0, 6), dtype=torch.float32)
            else:
                tracker_input = torch.tensor(
                    current_frame_dets[["x1", "y1", "x2", "y2", "confidence", "class_id"]].to_numpy(np.float32),
                    dtype=torch.float32,
                )
            tracked = tracker.update(tracker_input, None)
            for row in tracked:
                results.append({
                    "frame": frame_number, "x1": float(row[0]), "y1": float(row[1]),
                    "x2": float(row[2]), "y2": float(row[3]), "track_id": int(row[4]),
                    "class_id": int(row[5]), "confidence": float(row[6]),
                })
            if (frame_number - first_frame + 1) % 500 == 0:
                print(f"Processed frame {frame_number} / {last_frame}")
    finally:
        strack.multi_predict = staticmethod(original_multi_predict)
        strack.update = original_update
        strack.re_activate = original_re_activate

    tracked_df = pd.DataFrame(results, columns=[
        "frame", "x1", "y1", "x2", "y2", "track_id", "class_id", "confidence"
    ])
    if tracked_df.empty:
        raise ValueError("ByteTrack produced no tracked detections.")
    event_df = pd.DataFrame(events)
    if event_df.empty:
        raise ValueError("No valid pre-update prediction/match events were captured.")
    if event_df.duplicated(["frame", "track_id"]).any():
        raise ValueError("Duplicate residual events for a (frame, track_id) pair.")
    if not np.isfinite(event_df[["prediction_residual_pixels", "normalized_residual"]].to_numpy(float)).all():
        raise ValueError("Residual events contain non-finite residual values.")

    tracked_df["x"] = (tracked_df.x1 + tracked_df.x2) / 2.0
    tracked_df["y"] = (tracked_df.y1 + tracked_df.y2) / 2.0
    tracked_df = tracked_df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    grouped = tracked_df.groupby("track_id", sort=False)
    prev_x, prev_y, prev_frame = grouped.x.shift(1), grouped.y.shift(1), grouped.frame.shift(1)
    delta = tracked_df.frame - prev_frame
    tracked_df["vx"] = (tracked_df.x - prev_x) / delta
    tracked_df["vy"] = (tracked_df.y - prev_y) / delta
    tracked_df.loc[delta.isna(), ["vx", "vy"]] = 0.0

    trajectory = tracked_df.merge(
        event_df[["frame", "track_id", "prediction_residual_pixels", "normalized_residual",
                  "predicted_height", "gap_frames_since_last_match", "gap_group"]],
        on=["frame", "track_id"], how="left", validate="one_to_one"
    )
    trajectory = trajectory[[
        "frame", "track_id", "x", "y", "vx", "vy", "prediction_residual_pixels",
        "normalized_residual", "predicted_height", "gap_frames_since_last_match", "gap_group", "confidence"
    ]].sort_values(["frame", "track_id"]).reset_index(drop=True)

    summary_rows = [summarize_group(event_df, "all", "all_events")]
    for gap_group in ("consecutive_k0", "after_gap_k_positive", "first_observed_match"):
        subset = event_df[event_df.gap_group == gap_group]
        summary_rows.append(summarize_group(subset, "all", gap_group))
    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracked_df[["frame", "x1", "y1", "x2", "y2", "track_id", "class_id", "confidence"]].to_csv(TRACKED_CSV, index=False)
    event_df.to_csv(RESIDUAL_CSV, index=False)
    trajectory.to_csv(TRAJECTORY_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    saved_traj = pd.read_csv(TRAJECTORY_CSV)
    finite = int(saved_traj.prediction_residual_pixels.notna().sum())
    nonzero = int((saved_traj.prediction_residual_pixels.dropna() > 0).sum())
    nan_count = int(saved_traj.prediction_residual_pixels.isna().sum())
    if finite == 0 or nonzero == 0:
        raise AssertionError("Residual validation failed: no finite, nonzero residuals.")
    if saved_traj.duplicated(["frame", "track_id"]).any():
        raise AssertionError("Saved trajectory contains duplicate (frame, track_id) rows.")

    print("\nDAY 19 — LIGHT B0 RESIDUAL LOGGING")
    print("=" * 58)
    print(f"Parameters:                    {TRACK_THRESH}, {TRACK_BUFFER}, {MATCH_THRESH}, {FRAME_RATE} FPS")
    print(f"Tracked rows:                  {len(tracked_df)}")
    print(f"Unique track IDs:              {tracked_df.track_id.nunique()}")
    print(f"Prediction/match events:       {len(event_df)}")
    print(f"Finite trajectory residuals:   {finite}")
    print(f"Nonzero trajectory residuals:  {nonzero}")
    print(f"NaN trajectory residuals:      {nan_count}")
    print("\nResidual summary:")
    print(summary.to_string(index=False))
    print("\nSaved outputs:")
    for path in (TRACKED_CSV, RESIDUAL_CSV, TRAJECTORY_CSV, SUMMARY_CSV):
        print(path)
    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()
