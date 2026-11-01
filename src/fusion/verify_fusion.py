
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPATIBILITY_COLUMNS = [
    "frame",
    "fused_x",
    "fused_y",
    "fused_confidence",
    "source",
    "rgb_x",
    "rgb_y",
    "thermal_x",
    "thermal_y",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()

    fused = pd.read_csv(args.fused)
    prepared = pd.read_csv(args.prepared)

    if list(fused.columns[:9]) != COMPATIBILITY_COLUMNS:
        raise AssertionError(
            "Adaptive fused CSV does not start with the required nine columns."
        )

    if len(fused) != len(prepared):
        raise AssertionError(
            f"Row count mismatch: fused={len(fused)}, prepared={len(prepared)}"
        )

    required_prepared = {
        "frame", "x1", "y1", "x2", "y2", "confidence", "source"
    }
    missing = required_prepared - set(prepared.columns)
    if missing:
        raise ValueError(f"Prepared CSV missing columns: {sorted(missing)}")

    if not fused["frame"].equals(prepared["frame"]):
        raise AssertionError("Frame order/values differ between the two CSVs.")

    if not fused["source"].equals(prepared["source"]):
        raise AssertionError("Source labels differ between the two CSVs.")

    comparisons = [
        ("fused_x1", "x1"),
        ("fused_y1", "y1"),
        ("fused_x2", "x2"),
        ("fused_y2", "y2"),
    ]

    print("\n" + "=" * 62)
    print("DAY 12 - ADAPTIVE BOX COMPATIBILITY AUDIT")
    print("=" * 62)
    print(f"Fused rows:    {len(fused)}")
    print(f"Prepared rows: {len(prepared)}")

    for fused_col, prepared_col in comparisons:
        if fused_col not in fused.columns:
            raise ValueError(f"Fused CSV missing diagnostic column: {fused_col}")

        left = fused[fused_col].to_numpy(dtype=float)
        right = prepared[prepared_col].to_numpy(dtype=float)

        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise AssertionError(f"Non-finite values in {fused_col}/{prepared_col}")

        difference = np.abs(left - right)
        print(
            f"{fused_col} vs {prepared_col}: "
            f"max_abs_diff={difference.max(initial=0):.6f}px"
        )

        if not np.allclose(left, right, atol=1e-5, rtol=0):
            raise AssertionError(
                f"Prepared boxes do not match adaptive box coordinates: "
                f"{fused_col} vs {prepared_col}"
            )

    # Confidence must remain the value passed through to ByteTrack.
    confidence_diff = np.abs(
        fused["fused_confidence"].to_numpy(float)
        - prepared["confidence"].to_numpy(float)
    )
    print(
        "Confidence max difference: "
        f"{confidence_diff.max(initial=0):.8f}"
    )

    if not np.allclose(confidence_diff, 0, atol=1e-8):
        raise AssertionError("Prepared confidence differs from fused confidence.")

    print("\nBYTETrack BOX COMPATIBILITY AUDIT: PASS")


if __name__ == "__main__":
    main()
