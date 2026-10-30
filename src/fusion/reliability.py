
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "reliability_scores.csv"

REQUIRED_COLUMNS = ["frame", "x", "y", "confidence"]

DEFAULT_WINDOW = 30
DEFAULT_ALPHA = 1.5
DEFAULT_BETA = 1.0


def load_detections(csv_path, modality):
    """
    Load a detection CSV and retain valid detections.

    A valid detection must have:
    - A non-negative integer frame number.
    - Finite x and y coordinates.
    - Finite confidence greater than zero.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{modality} detection CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{modality} CSV is missing required columns: {missing}"
        )

    # Frame numbers must be valid.
    df["frame"] = pd.to_numeric(df["frame"], errors="raise")

    if df["frame"].isna().any():
        raise ValueError(
            f"{modality} CSV contains missing frame numbers."
        )

    if not np.isfinite(df["frame"]).all():
        raise ValueError(
            f"{modality} CSV contains non-finite frame numbers."
        )

    if (
        (df["frame"] < 0).any()
        or not np.equal(df["frame"], np.floor(df["frame"])).all()
    ):
        raise ValueError(
            f"{modality} CSV must contain non-negative integer frames."
        )

    df["frame"] = df["frame"].astype(int)

    # Convert detection measurements to numeric values.
    for column in ("x", "y", "confidence"):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    valid_mask = (
        df["x"].notna()
        & df["y"].notna()
        & df["confidence"].notna()
        & np.isfinite(df["x"])
        & np.isfinite(df["y"])
        & np.isfinite(df["confidence"])
        & (df["confidence"] > 0)
    )

    valid_df = df.loc[valid_mask].copy()

    return df, valid_df


def calculate_modality_reliability(
    valid_df,
    total_frames,
    window=DEFAULT_WINDOW,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
):
    """
    Calculate rolling detection rate, mean confidence,
    and reliability for one modality.

    The rolling window includes the current frame and
    contains at most `window` frames.

    For startup frames, only the available frames are used.
    """

    if window < 1:
        raise ValueError("Window size must be at least 1.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive.")

    if total_frames < 1:
        raise ValueError("Total frame count must be at least 1.")

    # Per-frame statistics.
    detection_count = np.bincount(
        valid_df["frame"].to_numpy(dtype=int),
        minlength=total_frames
    )

    confidence_sum = np.bincount(
        valid_df["frame"].to_numpy(dtype=int),
        weights=valid_df["confidence"].to_numpy(dtype=float),
        minlength=total_frames
    )

    # A frame contributes either 0 or 1 to detection coverage,
    # regardless of how many detections it contains.
    has_detection = (detection_count > 0).astype(float)

    # Prefix sums allow efficient rolling-window calculations.
    detection_prefix = np.concatenate(
        ([0.0], np.cumsum(has_detection))
    )

    confidence_prefix = np.concatenate(
        ([0.0], np.cumsum(confidence_sum))
    )

    count_prefix = np.concatenate(
        ([0], np.cumsum(detection_count))
    )

    detection_rates = np.zeros(total_frames, dtype=float)
    mean_confidences = np.zeros(total_frames, dtype=float)
    reliabilities = np.zeros(total_frames, dtype=float)

    for frame in range(total_frames):

        # Last `window` frames, including the current frame.
        start_frame = max(0, frame - window + 1)

        actual_window_size = frame - start_frame + 1

        frames_with_detection = (
            detection_prefix[frame + 1]
            - detection_prefix[start_frame]
        )

        confidence_total = (
            confidence_prefix[frame + 1]
            - confidence_prefix[start_frame]
        )

        valid_detection_total = (
            count_prefix[frame + 1]
            - count_prefix[start_frame]
        )

        # Step 1: Detection rate.
        detection_rate = (
            frames_with_detection / actual_window_size
        )

        # Step 2: Mean confidence.
        if valid_detection_total > 0:
            mean_confidence = (
                confidence_total / valid_detection_total
            )
        else:
            mean_confidence = 0.0

        # Step 3: Reliability.
        if detection_rate == 0.0 or mean_confidence == 0.0:
            reliability = 0.0
        else:
            reliability = (
                detection_rate ** alpha
                * mean_confidence ** beta
            )

        detection_rates[frame] = detection_rate
        mean_confidences[frame] = mean_confidence
        reliabilities[frame] = reliability

    return (
        detection_rates,
        mean_confidences,
        reliabilities
    )


def calculate_reliability(
    rgb_csv,
    ir_csv,
    output_csv=DEFAULT_OUTPUT,
    window=DEFAULT_WINDOW,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
):
    """
    Calculate RGB and IR rolling reliability scores
    and save them in a single CSV.
    """

    rgb_all, rgb_valid = load_detections(
        rgb_csv,
        "RGB"
    )

    ir_all, ir_valid = load_detections(
        ir_csv,
        "IR"
    )

    if rgb_all.empty and ir_all.empty:
        raise ValueError(
            "Both input CSVs are empty. No frame range can be determined."
        )

    # Process the complete frame sequence from frame 0
    # through the largest frame number in either CSV.
    max_frame = max(
        rgb_all["frame"].max() if not rgb_all.empty else 0,
        ir_all["frame"].max() if not ir_all.empty else 0,
    )

    total_frames = int(max_frame) + 1

    # Calculate reliability independently for each modality.
    (
        rgb_detection_rate,
        rgb_mean_conf,
        rgb_reliability,
    ) = calculate_modality_reliability(
        rgb_valid,
        total_frames,
        window,
        alpha,
        beta,
    )

    (
        ir_detection_rate,
        ir_mean_conf,
        ir_reliability,
    ) = calculate_modality_reliability(
        ir_valid,
        total_frames,
        window,
        alpha,
        beta,
    )

    # Construct the combined output.
    result = pd.DataFrame({
        "frame": np.arange(total_frames),

        "rgb_detection_rate": rgb_detection_rate,
        "ir_detection_rate": ir_detection_rate,

        "rgb_mean_conf": rgb_mean_conf,
        "ir_mean_conf": ir_mean_conf,

        "rgb_reliability": rgb_reliability,
        "ir_reliability": ir_reliability,
    })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(output_csv, index=False)

    # Report input statistics.
    print("\n" + "=" * 65)
    print("DAY 9 - ROLLING RELIABILITY ESTIMATOR")
    print("=" * 65)

    print(f"RGB input rows          : {len(rgb_all)}")
    print(f"RGB valid detections    : {len(rgb_valid)}")

    print(f"IR input rows           : {len(ir_all)}")
    print(f"IR valid detections     : {len(ir_valid)}")

    print(f"Frames processed        : {total_frames}")
    print(f"Window size (W)         : {window}")
    print(f"Alpha                   : {alpha}")
    print(f"Beta                    : {beta}")

    print("\nReliability statistics:")

    for modality, values in [
        ("RGB", rgb_reliability),
        ("IR", ir_reliability),
    ]:
        print(f"\n{modality}:")
        print(f"  Mean : {np.mean(values):.6f}")
        print(f"  Min  : {np.min(values):.6f}")
        print(f"  Max  : {np.max(values):.6f}")

    print("\nOutput preview:")
    print(result.head(10).to_string(index=False))

    print(f"\nReliability CSV saved to:\n{output_csv}")

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate rolling RGB and IR reliability."
    )

    parser.add_argument(
        "--rgb",
        type=Path,
        required=True,
        help="Path to RGB detection CSV."
    )

    parser.add_argument(
        "--ir",
        type=Path,
        required=True,
        help="Path to calibrated IR detection CSV."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output reliability CSV path."
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Rolling window size."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Detection-rate exponent."
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Mean-confidence exponent."
    )

    args = parser.parse_args()

    calculate_reliability(
        rgb_csv=args.rgb,
        ir_csv=args.ir,
        output_csv=args.output,
        window=args.window,
        alpha=args.alpha,
        beta=args.beta,
    )