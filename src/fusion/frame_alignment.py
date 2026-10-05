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
        Detection dataframe.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing required columns:\n"
            f"{missing_columns}"
        )

    return df


def group_detections_by_frame(df):
    """
    Group all detections belonging to each frame.

    Multiple people can be detected in the same frame,
    so frame number alone must not be treated as a unique row.

    Returns
    -------
    dict
        Dictionary where:

        key   = frame number
        value = dataframe containing all detections
                for that frame
    """

    grouped = {}

    for frame_number, frame_df in df.groupby("frame"):

        grouped[frame_number] = (
            frame_df
            .reset_index(drop=True)
            .copy()
        )

    return grouped


def align_frames(rgb_df, thermal_df):
    """
    Align RGB and thermal detections by frame number
    while preserving multiple detections per frame.

    No individual RGB-to-thermal person matching is
    performed at this stage.

    Returns
    -------
    dict

        {
            frame_number: {
                "rgb": DataFrame,
                "thermal": DataFrame
            }
        }

    Only frame numbers present in at least one modality
    are included.
    """

    rgb_by_frame = group_detections_by_frame(rgb_df)
    thermal_by_frame = group_detections_by_frame(thermal_df)

    all_frames = sorted(
        set(rgb_by_frame.keys()) |
        set(thermal_by_frame.keys())
    )

    aligned_frames = {}

    for frame_number in all_frames:

        rgb_detections = rgb_by_frame.get(
            frame_number,
            pd.DataFrame(columns=REQUIRED_COLUMNS)
        )

        thermal_detections = thermal_by_frame.get(
            frame_number,
            pd.DataFrame(columns=REQUIRED_COLUMNS)
        )

        aligned_frames[frame_number] = {
            "rgb": rgb_detections,
            "thermal": thermal_detections
        }

    return aligned_frames


def load_and_align(rgb_csv, thermal_csv):
    """
    Load RGB and thermal CSV files and align them
    by frame number.

    Returns
    -------
    dict
        Frame-aligned RGB and thermal detections.
    """

    rgb_df = load_detection_csv(rgb_csv)

    thermal_df = load_detection_csv(thermal_csv)

    aligned_frames = align_frames(
        rgb_df,
        thermal_df
    )

    return aligned_frames


def print_alignment_summary(aligned_frames):
    """
    Print a summary of the frame alignment.

    This is useful for verifying that multiple
    detections per frame are being preserved.
    """

    total_frames = len(aligned_frames)

    rgb_only = 0
    thermal_only = 0
    both_modalities = 0

    total_rgb_detections = 0
    total_thermal_detections = 0

    for frame_number, data in aligned_frames.items():

        rgb = data["rgb"]
        thermal = data["thermal"]

        rgb_has_detection = (
            (rgb["confidence"] > 0).any()
            if not rgb.empty
            else False
        )

        thermal_has_detection = (
            (thermal["confidence"] > 0).any()
            if not thermal.empty
            else False
        )

        if rgb_has_detection and thermal_has_detection:
            both_modalities += 1

        elif rgb_has_detection:
            rgb_only += 1

        elif thermal_has_detection:
            thermal_only += 1

        total_rgb_detections += (
            rgb["confidence"] > 0
        ).sum()

        total_thermal_detections += (
            thermal["confidence"] > 0
        ).sum()

    print()
    print("=" * 60)
    print("FRAME ALIGNMENT SUMMARY")
    print("=" * 60)

    print(f"Total unique frames       : {total_frames}")
    print(f"Frames with RGB + Thermal : {both_modalities}")
    print(f"RGB-only frames           : {rgb_only}")
    print(f"Thermal-only frames       : {thermal_only}")

    print()
    print(f"Total RGB detections      : {total_rgb_detections}")
    print(f"Total Thermal detections  : {total_thermal_detections}")


if __name__ == "__main__":

    rgb_csv = "C:/thermal project/Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance/outputs/rgb_detections.csv"
    thermal_csv = "C:/thermal project/Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance/outputs/thermal_detections.csv"

    aligned_frames = load_and_align(
        rgb_csv,
        thermal_csv
    )

    print_alignment_summary(
        aligned_frames
    )

    print()
    print("=" * 60)
    print("SAMPLE FRAME ALIGNMENT")
    print("=" * 60)

    sample_frame = next(
        iter(aligned_frames)
    )

    sample = aligned_frames[sample_frame]

    print(f"\nFrame: {sample_frame}")

    print(
        f"\nRGB detections: "
        f"{len(sample['rgb'])}"
    )

    print(
        f"Thermal detections: "
        f"{len(sample['thermal'])}"
    )

    print("\nRGB:")
    print(sample["rgb"])

    print("\nThermal:")
    print(sample["thermal"])