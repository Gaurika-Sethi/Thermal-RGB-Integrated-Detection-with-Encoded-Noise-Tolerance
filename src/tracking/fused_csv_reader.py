
from pathlib import Path

import pandas as pd


def load_fused_detections(csv_path=None):

    project_root = Path(__file__).resolve().parents[2]

    if csv_path is None:
        csv_path = (
            project_root
            / "outputs"
            / "fused_detections.csv"
        )

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fused CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "frame",
        "fused_x",
        "fused_y",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Fused CSV is missing required columns: "
            f"{missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="raise",
    )

    df["fused_x"] = pd.to_numeric(
        df["fused_x"],
        errors="raise",
    )

    df["fused_y"] = pd.to_numeric(
        df["fused_y"],
        errors="raise",
    )

    if df["frame"].isna().any():
        raise ValueError(
            "Frame numbers cannot be missing."
        )

    if not df["frame"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError(
            "Frame numbers must be integers."
        )

    if (df["frame"] < 0).any():
        raise ValueError(
            "Frame numbers cannot be negative."
        )

    # Keep all detections; do not collapse multiple
    # people into one measurement per frame.
    df = df.sort_values(
        "frame",
        kind="stable",
    ).reset_index(drop=True)

    print("=" * 55)
    print("DAY 22 - FUSED CSV LOADING COMPLETE")
    print("=" * 55)

    print(f"Input file            : {csv_path}")
    print(f"Total detection rows  : {len(df)}")
    print(f"Unique frames         : {df['frame'].nunique()}")

    missing_positions = (
        df["fused_x"].isna()
        | df["fused_y"].isna()
    )

    print(
        f"Rows with measurements: "
        f"{(~missing_positions).sum()}"
    )

    print(
        f"Rows with missing x/y : "
        f"{missing_positions.sum()}"
    )

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    load_fused_detections()
