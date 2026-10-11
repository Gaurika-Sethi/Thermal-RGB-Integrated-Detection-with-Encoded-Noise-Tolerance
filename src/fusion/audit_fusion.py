
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "fused_detections.csv"

REQUIRED_COLUMNS = [
    "frame",
    "fused_x",
    "fused_y",
    "fused_confidence",
    "source",
    "match_distance",
    "rgb_x",
    "rgb_y",
    "rgb_confidence",
    "thermal_x",
    "thermal_y",
    "thermal_confidence",
]

DISTANCE_THRESHOLD = 150.0
TOLERANCE = 1e-6


def audit_fusion(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fused CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The fused CSV is empty.")

    checks = {}

    checks["Valid frame numbers"] = (
        df["frame"].notna().all()
        and np.isfinite(df["frame"]).all()
        and (df["frame"] >= 0).all()
        and np.equal(df["frame"], np.floor(df["frame"])).all()
    )

    valid_sources = {
        "both", "rgb_only", "thermal_only", "none"
    }

    checks["Valid source labels"] = (
        df["source"].isin(valid_sources).all()
    )

    # All rows with a valid measurement must have x and y.
    measured = df["source"] != "none"

    checks["Measured rows have fused coordinates"] = (
        df.loc[measured, ["fused_x", "fused_y"]]
        .notna()
        .all()
        .all()
    )

    # Rows marked none must not contain fabricated measurements.
    none_rows = df["source"] == "none"

    checks["None rows have blank fused coordinates"] = (
        df.loc[none_rows, ["fused_x", "fused_y"]]
        .isna()
        .all()
        .all()
    )

    # Verify the confidence-weighted formula for matched rows.
    both = df[df["source"] == "both"]

    if not both.empty:
        rc = both["rgb_confidence"].to_numpy(dtype=float)
        tc = both["thermal_confidence"].to_numpy(dtype=float)

        rx = both["rgb_x"].to_numpy(dtype=float)
        tx = both["thermal_x"].to_numpy(dtype=float)
        ry = both["rgb_y"].to_numpy(dtype=float)
        ty = both["thermal_y"].to_numpy(dtype=float)

        fx = both["fused_x"].to_numpy(dtype=float)
        fy = both["fused_y"].to_numpy(dtype=float)
        fc = both["fused_confidence"].to_numpy(dtype=float)

        total = rc + tc

        checks["Matched rows have positive confidence sum"] = (
            np.isfinite(total).all() and (total > 0).all()
        )

        if np.isfinite(total).all() and (total > 0).all():
            expected_x = (rc * rx + tc * tx) / total
            expected_y = (rc * ry + tc * ty) / total
            expected_conf = total / 2

            checks["Weighted x formula"] = np.allclose(
                fx, expected_x, atol=TOLERANCE, rtol=TOLERANCE
            )

            checks["Weighted y formula"] = np.allclose(
                fy, expected_y, atol=TOLERANCE, rtol=TOLERANCE
            )

            checks["Average confidence formula"] = np.allclose(
                fc, expected_conf, atol=TOLERANCE, rtol=TOLERANCE
            )
        else:
            checks["Weighted x formula"] = False
            checks["Weighted y formula"] = False
            checks["Average confidence formula"] = False

        checks["Matched rows have both modality coordinates"] = (
            both[
                [
                    "rgb_x", "rgb_y", "rgb_confidence",
                    "thermal_x", "thermal_y", "thermal_confidence",
                ]
            ]
            .notna()
            .all()
            .all()
        )

        checks["Matched distances within threshold"] = (
            both["match_distance"].notna().all()
            and np.isfinite(both["match_distance"]).all()
            and (both["match_distance"] >= 0).all()
            and (
                both["match_distance"]
                <= DISTANCE_THRESHOLD + TOLERANCE
            ).all()
        )
    else:
        print("WARNING: No matched rows available to verify the formula.")

    # Check single-modality rows preserve their original measurements.
    for source, prefix in [
        ("rgb_only", "rgb"),
        ("thermal_only", "thermal"),
    ]:
        rows = df[df["source"] == source]

        if rows.empty:
            checks[f"{source} coordinates preserved"] = True
            continue

        checks[f"{source} coordinates preserved"] = (
            np.allclose(
                rows["fused_x"].to_numpy(dtype=float),
                rows[f"{prefix}_x"].to_numpy(dtype=float),
                atol=TOLERANCE,
                rtol=TOLERANCE,
                equal_nan=False,
            )
            and np.allclose(
                rows["fused_y"].to_numpy(dtype=float),
                rows[f"{prefix}_y"].to_numpy(dtype=float),
                atol=TOLERANCE,
                rtol=TOLERANCE,
                equal_nan=False,
            )
            and np.allclose(
                rows["fused_confidence"].to_numpy(dtype=float),
                rows[f"{prefix}_confidence"].to_numpy(dtype=float),
                atol=TOLERANCE,
                rtol=TOLERANCE,
                equal_nan=False,
            )
        )

    print("\n" + "=" * 60)
    print("DAY 21 - FINAL FUSION AUDIT")
    print("=" * 60)

    print(f"CSV: {csv_path}")
    print(f"Total output rows: {len(df)}")
    print(f"Unique frames: {df['frame'].nunique()}")
    print(
        f"Frame range: {df['frame'].min()} "
        f"to {df['frame'].max()}"
    )

    print("\nSource distribution:")
    print(
        df["source"]
        .value_counts()
        .reindex(
            ["both", "rgb_only", "thermal_only", "none"],
            fill_value=0,
        )
        .to_string()
    )

    print("\nAudit checks:")
    for description, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")

    passed_count = sum(checks.values())
    total_checks = len(checks)

    print(f"\nResult: {passed_count}/{total_checks} checks passed.")

    if not all(checks.values()):
        print(
            "AUDIT NOT FULLY PASSED. Investigate failed checks "
            "before treating the CSV as validated."
        )
        return False

    print("All applicable audit checks passed.")
    print(
        "Note: These checks validate CSV consistency and formula "
        "arithmetic, not the real-world correctness of person matches."
    )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit the fused RGB-thermal detection CSV."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to fused_detections.csv",
    )

    args = parser.parse_args()

    success = audit_fusion(args.csv)

    if not success:
        raise SystemExit(1)
