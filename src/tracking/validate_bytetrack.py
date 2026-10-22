
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT / "outputs" / "bytetrack_detections.csv"
TRACKED_CSV = ROOT / "outputs" / "tracked_detections.csv"

REPORT_FILE = ROOT / "outputs" / "bytetrack_validation_report.txt"
TRACK_SUMMARY_FILE = ROOT / "outputs" / "track_summary.csv"
FRAME_SUMMARY_FILE = ROOT / "outputs" / "track_frame_summary.csv"


def main():
    for path in [INPUT_CSV, TRACKED_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    detections = pd.read_csv(INPUT_CSV)
    tracked = pd.read_csv(TRACKED_CSV)

    detection_cols = {
        "frame", "x1", "y1", "x2", "y2", "confidence", "class_id"
    }
    tracked_cols = detection_cols | {"track_id"}

    if not detection_cols.issubset(detections.columns):
        raise ValueError(
            f"Input CSV missing columns: {sorted(detection_cols - set(detections.columns))}"
        )

    if not tracked_cols.issubset(tracked.columns):
        raise ValueError(
            f"Tracked CSV missing columns: {sorted(tracked_cols - set(tracked.columns))}"
        )

    numeric_cols = [
        "frame", "x1", "y1", "x2", "y2", "confidence", "class_id"
    ]

    for col in numeric_cols:
        detections[col] = pd.to_numeric(detections[col], errors="coerce")
        tracked[col] = pd.to_numeric(tracked[col], errors="coerce")

    tracked["track_id"] = pd.to_numeric(
        tracked["track_id"], errors="coerce"
    )

    issues = []

    def check(condition, message):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {message}")
        if not condition:
            issues.append(message)

    # ---------------------------------------------------------
    # 1. Structural and numeric validation
    # ---------------------------------------------------------
    check(not detections.empty, "Input detections are not empty")
    check(not tracked.empty, "Tracked output is not empty")

    check(
        not tracked[numeric_cols + ["track_id"]].isna().any().any(),
        "Tracked output contains no missing numeric values",
    )

    if not tracked.empty:
        values = tracked[numeric_cols + ["track_id"]].to_numpy(dtype=float)
        check(
            np.isfinite(values).all(),
            "Tracked output contains only finite numeric values",
        )

        check(
            (tracked["x2"] > tracked["x1"]).all()
            and (tracked["y2"] > tracked["y1"]).all(),
            "All tracked bounding boxes have positive dimensions",
        )

        check(
            tracked["confidence"].between(0, 1).all(),
            "All tracked confidence values are in [0, 1]",
        )

        check(
            (tracked["track_id"] > 0).all()
            and (tracked["track_id"] % 1 == 0).all(),
            "Track IDs are positive integers",
        )

        check(
            (tracked["frame"] % 1 == 0).all(),
            "Tracked frame numbers are integers",
        )

        check(
            tracked["class_id"].eq(0).all(),
            "All tracked detections are person class (class_id 0)",
        )

    # ---------------------------------------------------------
    # 2. Frame coverage and output counts
    # ---------------------------------------------------------
    detections["frame"] = detections["frame"].astype(int)
    tracked["frame"] = tracked["frame"].astype(int)

    input_frames = set(detections["frame"].unique())
    tracked_frames = set(tracked["frame"].unique())

    check(
        tracked_frames.issubset(input_frames),
        "Every tracked frame exists in the input detections",
    )

    input_counts = detections.groupby("frame").size().rename("input_detections")
    track_counts = tracked.groupby("frame").size().rename("tracked_rows")

    frame_summary = pd.concat(
        [input_counts, track_counts], axis=1
    ).fillna(0).astype(int)

    frame_summary["retained_detections"] = (
        frame_summary["tracked_rows"] / frame_summary["input_detections"]
    ).replace([np.inf, -np.inf], np.nan)

    frame_summary["unique_track_ids"] = (
        tracked.groupby("frame")["track_id"].nunique()
    ).reindex(frame_summary.index).fillna(0).astype(int)

    # ---------------------------------------------------------
    # 3. Duplicate track IDs in a single frame
    # ---------------------------------------------------------
    duplicate_pairs = tracked.duplicated(
        subset=["frame", "track_id"], keep=False
    )
    duplicate_rows = int(duplicate_pairs.sum())

    check(
        duplicate_rows == 0,
        "No track ID appears more than once in the same frame",
    )

    # ---------------------------------------------------------
    # 4. Per-track lifetime and continuity
    # ---------------------------------------------------------
    track_records = []

    for track_id, group in tracked.groupby("track_id"):
        frames = np.sort(group["frame"].unique())
        gaps = np.diff(frames) - 1

        track_records.append({
            "track_id": int(track_id),
            "detections": len(group),
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "span_frames": int(frames[-1] - frames[0] + 1),
            "observed_frames": len(frames),
            "missing_frames_inside_span": int(gaps[gaps > 0].sum()),
            "largest_internal_gap": int(gaps.max()) if len(gaps) else 0,
            "mean_confidence": float(group["confidence"].mean()),
            "min_confidence": float(group["confidence"].min()),
            "max_confidence": float(group["confidence"].max()),
        })

    track_summary = pd.DataFrame(track_records)

    if not track_summary.empty:
        track_summary["observed_frame_ratio"] = (
            track_summary["observed_frames"] / track_summary["span_frames"]
        )

        short_tracks = int((track_summary["observed_frames"] < 3).sum())
        fragmented_tracks = int(
            (track_summary["missing_frames_inside_span"] > 0).sum()
        )

        check(
            short_tracks == 0,
            f"No very short tracks (fewer than 3 observed frames); found {short_tracks}",
        )

        # This is diagnostic rather than a correctness requirement.
        print(
            "[INFO] Tracks with internal frame gaps: "
            f"{fragmented_tracks}/{len(track_summary)}"
        )

    # ---------------------------------------------------------
    # 5. Compare input and tracked counts
    # ---------------------------------------------------------
    input_rows = len(detections)
    tracked_rows = len(tracked)
    retention = tracked_rows / input_rows if input_rows else 0

    print("\n" + "=" * 60)
    print("BYTETRACK VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Input detection rows       : {input_rows}")
    print(f"Tracked output rows        : {tracked_rows}")
    print(f"Output/input row ratio     : {retention:.2%}")
    print(f"Input unique frames        : {detections['frame'].nunique()}")
    print(f"Tracked unique frames      : {tracked['frame'].nunique()}")
    print(f"Unique track IDs           : {tracked['track_id'].nunique()}")
    print(f"Duplicate frame/ID rows    : {duplicate_rows}")

    if not track_summary.empty:
        print("\nTRACK LIFETIME STATISTICS (observed frames)")
        print(track_summary["observed_frames"].describe().to_string())
        print(
            "\nTracks with internal gaps : "
            f"{int((track_summary['missing_frames_inside_span'] > 0).sum())}"
        )
        print(
            "Tracks observed in < 3 frames: "
            f"{int((track_summary['observed_frames'] < 3).sum())}"
        )

    # ---------------------------------------------------------
    # 6. Save detailed reports
    # ---------------------------------------------------------
    ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    track_summary.to_csv(TRACK_SUMMARY_FILE, index=False)
    frame_summary.to_csv(FRAME_SUMMARY_FILE, index=True)
    
    with REPORT_FILE.open("w", encoding="utf-8") as report:
        report.write("BYTETRACK VALIDATION REPORT\n")
        report.write("=" * 60 + "\n")
        report.write(f"Input detections: {input_rows}\n")
        report.write(f"Tracked output rows: {tracked_rows}\n")
        report.write(f"Output/input ratio: {retention:.2%}\n")
        report.write(f"Input frames: {detections['frame'].nunique()}\n")
        report.write(f"Tracked frames: {tracked['frame'].nunique()}\n")
        report.write(f"Unique track IDs: {tracked['track_id'].nunique()}\n")
        report.write(f"Duplicate frame/ID rows: {duplicate_rows}\n")
        report.write(f"Failed checks: {len(issues)}\n\n")

        if issues:
            report.write("ISSUES\n")
            for issue in issues:
                report.write(f"- {issue}\n")
        else:
            report.write("All hard validation checks passed.\n")

        report.write("\nTrack lifetime statistics:\n")
        if not track_summary.empty:
            report.write(
                track_summary["observed_frames"].describe().to_string()
            )
            report.write("\n\nLongest tracks:\n")
            report.write(
                track_summary.sort_values(
                    "observed_frames", ascending=False
                ).head(10).to_string(index=False)
            )

    print("\nREPORTS SAVED")
    print(f"Text report : {REPORT_FILE}")
    print(f"Track stats : {TRACK_SUMMARY_FILE}")
    print(f"Frame stats : {FRAME_SUMMARY_FILE}")
    print(f"\nFailed hard checks: {len(issues)}")

    if issues:
        print("Result: INVESTIGATION REQUIRED")
    else:
        print("Result: STRUCTURAL VALIDATION PASSED")
        print("Note: Visual identity continuity still requires video review.")


if __name__ == "__main__":
    main()
