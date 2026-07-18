from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "frame",
    "x",
    "y",
    "width",
    "height",
    "confidence"
]


def load_detection_csv(csv_path):
    """
    Load a detection CSV and validate its schema.

    Parameters
    ----------
    csv_path : str or Path

    Returns
    -------
    pandas.DataFrame
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns:\n{missing}"
        )

    return df


def align_frames(rgb_df, thermal_df):
    """
    Align RGB and thermal detections by frame number.

    Returns
    -------
    pandas.DataFrame

    Columns:

    frame

    rgb_x
    rgb_y
    rgb_width
    rgb_height
    rgb_confidence

    thermal_x
    thermal_y
    thermal_width
    thermal_height
    thermal_confidence
    """

    rgb = rgb_df.rename(
        columns={
            "x": "rgb_x",
            "y": "rgb_y",
            "width": "rgb_width",
            "height": "rgb_height",
            "confidence": "rgb_confidence"
        }
    )

    thermal = thermal_df.rename(
        columns={
            "x": "thermal_x",
            "y": "thermal_y",
            "width": "thermal_width",
            "height": "thermal_height",
            "confidence": "thermal_confidence"
        }
    )

    aligned = pd.merge(
        rgb,
        thermal,
        on="frame",
        how="outer",
        sort=True
    )

    aligned = aligned.sort_values(
        by="frame"
    ).reset_index(drop=True)

    return aligned


def load_and_align(rgb_csv, thermal_csv):
    """
    Convenience function.

    Loads both CSVs and returns
    an aligned dataframe.
    """

    rgb_df = load_detection_csv(rgb_csv)

    thermal_df = load_detection_csv(thermal_csv)

    aligned_df = align_frames(
        rgb_df,
        thermal_df
    )

    return aligned_df


if __name__ == "__main__":

    rgb_csv = "rgb_detections.csv"
    thermal_csv = "thermal_detections.csv"

    aligned = load_and_align(
        rgb_csv,
        thermal_csv
    )

    print()

    print("=" * 60)
    print("FRAME ALIGNMENT COMPLETE")
    print("=" * 60)

    print()

    print(aligned.head())

    print()

    print(f"Aligned frames: {len(aligned)}")