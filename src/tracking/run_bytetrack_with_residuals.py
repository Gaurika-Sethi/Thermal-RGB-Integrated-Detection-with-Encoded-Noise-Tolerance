from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bytetracker import BYTETracker
from bytetracker import byte_tracker


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

INPUT_CSV = OUTPUTS / "bytetrack_detections.csv"
FUSED_CSV = OUTPUTS / "fused_detections_calibrated.csv"

TRACKED_CSV = OUTPUTS / "tracked_detections_test_e_residuals.csv"
RESIDUAL_CSV = OUTPUTS / "prediction_residuals_test_e.csv"
TRAJECTORY_CSV = OUTPUTS / "trajectory_test_e_with_residuals.csv"

TRACK_THRESH = 0.45
TRACK_BUFFER = 50
MATCH_THRESH = 0.90
FRAME_RATE = 30

# ByteTrack receives float32 detections; this tolerance allows only
# insignificant numeric conversion differences when identifying the input row.
BOX_MATCH_TOLERANCE = 1e-2
CONF_MATCH_TOLERANCE = 1e-4


def main():
    for path in [INPUT_CSV, FUSED_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    detections = pd.read_csv(INPUT_CSV)
    fused = pd.read_csv(FUSED_CSV)

    detection_columns = {
        "frame", "x1", "y1", "x2", "y2", "confidence", "class_id"
    }
    fused_columns = {"frame", "fused_x", "fused_y"}

    if not detection_columns.issubset(detections.columns):
        raise ValueError(
            f"Input CSV missing columns: "
            f"{sorted(detection_columns - set(detections.columns))}"
        )
    if not fused_columns.issubset(fused.columns):
        raise ValueError(
            f"Fused CSV missing columns: "
            f"{sorted(fused_columns - set(fused.columns))}"
        )
    if detections.empty or fused.empty:
        raise ValueError("Input detections and fused detections must be nonempty.")

    numeric_detection_columns = [
        "frame", "x1", "y1", "x2", "y2", "confidence", "class_id"
    ]
    for col in numeric_detection_columns:
        detections[col] = pd.to_numeric(detections[col], errors="raise")
    for col in ["frame", "fused_x", "fused_y"]:
        fused[col] = pd.to_numeric(fused[col], errors="raise")

    if not np.isfinite(
        detections[numeric_detection_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Input detections contain non-finite values.")
    if not np.isfinite(
        fused[["frame", "fused_x", "fused_y"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Fused detections contain non-finite values.")

    for name, frame_series in [
        ("input detections", detections["frame"]),
        ("fused detections", fused["frame"]),
    ]:
        if not frame_series.apply(lambda value: float(value).is_integer()).all():
            raise ValueError(f"{name} frame numbers must be integers.")

    detections["frame"] = detections["frame"].astype(int)
    detections["class_id"] = detections["class_id"].astype(int)
    fused["frame"] = fused["frame"].astype(int)

    # The input-preparation script emits one ByteTrack row per fused row in
    # the same order, within each frame. Verify that assumption before relying
    # on positional correspondence.
    detection_counts = detections.groupby("frame", sort=True).size()
    fused_counts = fused.groupby("frame", sort=True).size()
    if not detection_counts.equals(fused_counts):
        raise ValueError(
            "Per-frame row counts differ between ByteTrack input and calibrated "
            "fusion CSV; cannot preserve row correspondence safely."
        )

    detection_frames = detections["frame"].drop_duplicates().tolist()
    fused_frames = fused["frame"].drop_duplicates().tolist()
    if detection_frames != fused_frames:
        raise ValueError(
            "Frame ordering differs between ByteTrack input and calibrated fusion CSV."
        )

    detections = detections.reset_index(drop=True)
    fused = fused.reset_index(drop=True)
    detections["_input_detection_row_id"] = np.arange(len(detections))
    fused["_fused_row_id"] = np.arange(len(fused))

    events = []
    results = []

    current_frame = None
    current_frame_dets = None
    current_frame_fused = None
    used_detection_positions = set()

    strack = byte_tracker.STrack
    original_multi_predict = strack.multi_predict
    original_update = strack.update
    original_re_activate = strack.re_activate

    def instrumented_multi_predict(stracks):
        result = original_multi_predict(stracks)

        # ByteTrack has just advanced the Kalman state and has not yet
        # performed the detection-association update for this frame.
        for track in stracks:
            if track.mean is not None:
                track._residual_predicted_center = np.asarray(
                    track.mean[:2], dtype=float
                ).copy()
                track._residual_prediction_frame = current_frame
        return result

    def find_input_detection_position(new_track):
        if current_frame_dets is None or current_frame_fused is None:
            raise RuntimeError("Current-frame detections were not initialized.")

        # In this installed bytetracker version, BYTETracker.update()
        # converts xyxy input boxes to center-based xywh before constructing
        # STrack objects. For a newly constructed detection, _tlwh therefore
        # contains [center_x, center_y, width, height] despite its name.
        # Use that raw stored representation, not the .tlwh property name.
        new_xywh = np.asarray(new_track._tlwh, dtype=float)
        if new_xywh.shape != (4,) or not np.isfinite(new_xywh).all():
            raise RuntimeError("Matched ByteTrack detection has invalid stored box values.")

        frame_dets = current_frame_dets
        input_xywh = np.column_stack([
            (frame_dets["x1"] + frame_dets["x2"]).to_numpy(dtype=float) / 2.0,
            (frame_dets["y1"] + frame_dets["y2"]).to_numpy(dtype=float) / 2.0,
            (frame_dets["x2"] - frame_dets["x1"]).to_numpy(dtype=float),
            (frame_dets["y2"] - frame_dets["y1"]).to_numpy(dtype=float),
        ])
        box_errors = np.max(np.abs(input_xywh - new_xywh[None, :]), axis=1)

        score = float(new_track.score)
        score_errors = np.abs(
            frame_dets["confidence"].to_numpy(dtype=float) - score
        )
        class_id = int(new_track.cls)
        class_matches = frame_dets["class_id"].to_numpy(dtype=int) == class_id

        available = np.ones(len(frame_dets), dtype=bool)
        if used_detection_positions:
            available[list(used_detection_positions)] = False

        candidates = np.flatnonzero(
            available
            & class_matches
            & (box_errors <= BOX_MATCH_TOLERANCE)
            & (score_errors <= CONF_MATCH_TOLERANCE)
        )

        if len(candidates) != 1:
            # Never silently choose a potentially wrong fused detection.
            raise RuntimeError(
                f"Frame {current_frame}: expected one input row for a matched "
                f"ByteTrack detection, found {len(candidates)}. "
                f"Best center-xywh error={float(box_errors.min()):.6f}, "
                f"best confidence error={float(score_errors.min()):.6f}."
            )

        position = int(candidates[0])
        used_detection_positions.add(position)
        return position

    def record_match(track, new_track, operation):
        prediction_frame = getattr(
            track, "_residual_prediction_frame", None
        )
        predicted = getattr(
            track, "_residual_predicted_center", None
        )

        # A new/unconfirmed track can be updated without a prediction in this
        # frame. Do not report a stale prediction as a current-frame residual.
        if prediction_frame != current_frame or predicted is None:
            return
        if not np.isfinite(predicted).all():
            return

        # Recover the exact input row from its box and confidence, then use
        # that row's verified positional counterpart in the calibrated fusion
        # CSV. This avoids post-hoc nearest-centroid rematching.
        input_position = find_input_detection_position(new_track)
        input_row = current_frame_dets.iloc[input_position]
        fused_row = current_frame_fused.iloc[input_position]

        # Compute the detection center from the exact matched ByteTrack input
        # row. Do not use new_track.tlwh here: in this installed bytetracker
        # version, a fresh detection's internal _tlwh stores center-based
        # [cx, cy, width, height], while the .tlwh property is not a reliable
        # source for interpreting that fresh detection's center.
        detection_x = float((input_row["x1"] + input_row["x2"]) / 2.0)
        detection_y = float((input_row["y1"] + input_row["y2"]) / 2.0)
        fused_x = float(fused_row["fused_x"])
        fused_y = float(fused_row["fused_y"])
        residual = float(np.hypot(
            fused_x - float(predicted[0]),
            fused_y - float(predicted[1]),
        ))

        events.append({
            "frame": int(current_frame),
            "track_id": int(track.track_id),
            "input_detection_row_id": int(input_row["_input_detection_row_id"]),
            "fused_row_id": int(fused_row["_fused_row_id"]),
            "predicted_x": float(predicted[0]),
            "predicted_y": float(predicted[1]),
            "detection_center_x": detection_x,
            "detection_center_y": detection_y,
            "fused_x": fused_x,
            "fused_y": fused_y,
            "detection_to_fused_distance": float(np.hypot(
                detection_x - fused_x, detection_y - fused_y
            )),
            "prediction_residual_pixels": residual,
            "association": operation,
        })

    def instrumented_update(track, new_track, frame_id):
        record_match(track, new_track, "update")
        return original_update(track, new_track, frame_id)

    def instrumented_re_activate(track, new_track, frame_id, new_id=False):
        record_match(track, new_track, "re_activate")
        return original_re_activate(
            track, new_track, frame_id, new_id=new_id
        )

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

        first_frame = int(detections["frame"].min())
        last_frame = int(detections["frame"].max())

        for frame_number in range(first_frame, last_frame + 1):
            current_frame = frame_number
            current_frame_dets = (
                detections[detections["frame"] == frame_number]
                .reset_index(drop=True)
            )
            current_frame_fused = (
                fused[fused["frame"] == frame_number]
                .reset_index(drop=True)
            )
            used_detection_positions = set()

            if len(current_frame_dets) != len(current_frame_fused):
                raise RuntimeError(
                    f"Frame {frame_number}: input/fusion row counts differ."
                )

            if current_frame_dets.empty:
                tracker_input = torch.empty((0, 6), dtype=torch.float32)
            else:
                tracker_input = torch.tensor(
                    current_frame_dets[
                        ["x1", "y1", "x2", "y2", "confidence", "class_id"]
                    ].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )

            tracked = tracker.update(tracker_input, None)

            for row in tracked:
                results.append({
                    "frame": frame_number,
                    "x1": float(row[0]),
                    "y1": float(row[1]),
                    "x2": float(row[2]),
                    "y2": float(row[3]),
                    "track_id": int(row[4]),
                    "class_id": int(row[5]),
                    "confidence": float(row[6]),
                })

            if (frame_number - first_frame + 1) % 50 == 0:
                print(f"Processed frame {frame_number} / {last_frame}")

    finally:
        strack.multi_predict = staticmethod(original_multi_predict)
        strack.update = original_update
        strack.re_activate = original_re_activate

    tracked_df = pd.DataFrame(results, columns=[
        "frame", "x1", "y1", "x2", "y2",
        "track_id", "class_id", "confidence",
    ])
    if tracked_df.empty:
        raise ValueError("ByteTrack produced no tracked detections.")

    event_df = pd.DataFrame(events)
    if event_df.empty:
        raise ValueError("No valid pre-update prediction/match events captured.")
    if event_df.duplicated(["frame", "track_id"]).any():
        raise ValueError("Duplicate prediction events for a frame/track ID.")
    if not np.isfinite(
        event_df[["predicted_x", "predicted_y", "fused_x", "fused_y",
                  "prediction_residual_pixels"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Residual events contain non-finite numeric values.")

    # Preserve the same trajectory center/velocity logic as the existing
    # trajectory generator, but write to a separate Test E output.
    tracked_df["x"] = (tracked_df["x1"] + tracked_df["x2"]) / 2.0
    tracked_df["y"] = (tracked_df["y1"] + tracked_df["y2"]) / 2.0
    tracked_df = tracked_df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    grouped = tracked_df.groupby("track_id", sort=False)
    previous_x = grouped["x"].shift(1)
    previous_y = grouped["y"].shift(1)
    previous_frame = grouped["frame"].shift(1)
    frame_delta = tracked_df["frame"] - previous_frame

    tracked_df["vx"] = (tracked_df["x"] - previous_x) / frame_delta
    tracked_df["vy"] = (tracked_df["y"] - previous_y) / frame_delta
    tracked_df.loc[frame_delta.isna(), ["vx", "vy"]] = 0.0

    trajectory = tracked_df.merge(
        event_df[["frame", "track_id", "prediction_residual_pixels"]],
        on=["frame", "track_id"],
        how="left",
        validate="one_to_one",
    )
    trajectory_columns = [
        "frame", "track_id", "x", "y", "vx", "vy",
        "prediction_residual_pixels", "confidence",
    ]
    trajectory = trajectory[trajectory_columns].sort_values(
        ["frame", "track_id"]
    ).reset_index(drop=True)

    residual_values = trajectory["prediction_residual_pixels"].to_numpy(dtype=float)
    finite_count = int(np.isfinite(residual_values).sum())
    nonzero_count = int(
        (np.isfinite(residual_values) & (residual_values > 0)).sum()
    )
    nan_count = int(np.isnan(residual_values).sum())

    if finite_count == 0 or nonzero_count == 0:
        raise AssertionError(
            "Residual verification failed: no finite, nonzero residuals."
        )

    # Save only after computation and validation succeed.
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    tracked_df[
        ["frame", "x1", "y1", "x2", "y2",
         "track_id", "class_id", "confidence"]
    ].to_csv(TRACKED_CSV, index=False)

    residual_columns = [
        "frame", "track_id", "input_detection_row_id", "fused_row_id",
        "predicted_x", "predicted_y", "detection_center_x",
        "detection_center_y", "fused_x", "fused_y",
        "detection_to_fused_distance", "prediction_residual_pixels",
        "association",
    ]
    event_df[residual_columns].to_csv(RESIDUAL_CSV, index=False)
    trajectory.to_csv(TRAJECTORY_CSV, index=False)

    # Validate the saved files, not just in-memory data.
    saved_trajectory = pd.read_csv(TRAJECTORY_CSV)
    saved_residuals = pd.read_csv(RESIDUAL_CSV)
    assert "prediction_residual_pixels" in saved_trajectory.columns
    assert len(saved_trajectory) == len(tracked_df)
    assert not saved_trajectory.duplicated(["frame", "track_id"]).any()

    saved_values = saved_trajectory["prediction_residual_pixels"].to_numpy(dtype=float)
    saved_finite = int(np.isfinite(saved_values).sum())
    saved_nonzero = int(
        (np.isfinite(saved_values) & (saved_values > 0)).sum()
    )
    if saved_finite == 0 or saved_nonzero == 0:
        raise AssertionError("Saved residual verification failed.")

    print("\nTEST E - PREDICTION RESIDUAL LOGGING COMPLETE")
    print("=" * 58)
    print(f"Parameters:                   {TRACK_THRESH}, {TRACK_BUFFER}, {MATCH_THRESH}")
    print(f"Tracked rows:                 {len(tracked_df)}")
    print(f"Unique track IDs:             {tracked_df['track_id'].nunique()}")
    print(f"Prediction/match events:      {len(saved_residuals)}")
    print(f"Finite trajectory residuals:  {saved_finite}")
    print(f"Nonzero trajectory residuals: {saved_nonzero}")
    print(f"NaN trajectory residuals:     {int(np.isnan(saved_values).sum())}")
    print(f"Tracked output:               {TRACKED_CSV}")
    print(f"Residual log:                 {RESIDUAL_CSV}")
    print(f"Trajectory output:            {TRAJECTORY_CSV}")
    print("Validation:                   PASSED")


if __name__ == "__main__":
    main()
