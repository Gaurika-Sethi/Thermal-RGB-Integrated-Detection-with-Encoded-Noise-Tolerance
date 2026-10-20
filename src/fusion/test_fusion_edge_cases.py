
import pandas as pd

from fusion_pipeline import fuse_frame_all_persons


COLUMNS = ["x", "y", "confidence"]


def detections(rows):
    """Create a detection DataFrame from (x, y, confidence) tuples."""
    return pd.DataFrame(rows, columns=COLUMNS)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main():
    # Case 1: Both modalities have no detections.
    result = fuse_frame_all_persons(
        10,
        detections([]),
        detections([]),
    )

    check(len(result) == 1, "Both empty -> one placeholder row")
    check(result[0]["source"] == "none", "Both empty -> source=none")
    check(
        pd.isna(result[0]["fused_x"])
        and pd.isna(result[0]["fused_y"]),
        "Both empty -> fused coordinates remain blank",
    )

    # Case 2: RGB empty, thermal has two detections.
    thermal = detections([
        (100.0, 120.0, 0.8),
        (300.0, 220.0, 0.6),
    ])

    result = fuse_frame_all_persons(
        11,
        detections([]),
        thermal,
    )

    check(len(result) == 2, "RGB empty -> preserve both thermal detections")
    check(
        all(row["source"] == "thermal_only" for row in result),
        "RGB empty -> source=thermal_only",
    )
    check(
        all(pd.isna(row["rgb_x"]) for row in result),
        "RGB empty -> RGB coordinates remain blank",
    )

    # Case 3: Thermal empty, RGB has two detections.
    rgb = detections([
        (110.0, 130.0, 0.9),
        (400.0, 350.0, 0.7),
    ])

    result = fuse_frame_all_persons(
        12,
        rgb,
        detections([]),
    )

    check(len(result) == 2, "Thermal empty -> preserve both RGB detections")
    check(
        all(row["source"] == "rgb_only" for row in result),
        "Thermal empty -> source=rgb_only",
    )
    check(
        all(pd.isna(row["thermal_x"]) for row in result),
        "Thermal empty -> thermal coordinates remain blank",
    )

    # Case 4: Invalid detections are filtered.
    invalid_rgb = detections([
        (100.0, 100.0, 0.0),
        (float("nan"), 100.0, 0.8),
    ])

    result = fuse_frame_all_persons(
        13,
        invalid_rgb,
        detections([]),
    )

    check(
        len(result) == 1 and result[0]["source"] == "none",
        "Invalid RGB detections -> treated as no valid detections",
    )

    # Case 5: One matched pair and one unmatched RGB detection.
    rgb = detections([
        (100.0, 100.0, 0.8),
        (700.0, 700.0, 0.9),
    ])
    thermal = detections([
        (105.0, 105.0, 0.6),
    ])

    result = fuse_frame_all_persons(14, rgb, thermal)

    sources = [row["source"] for row in result]

    check("both" in sources, "Nearby detections are fused")
    check("rgb_only" in sources, "Unmatched RGB detection is retained")
    check(len(result) == 2, "No detection is duplicated or discarded")

    print("\nAll edge-case tests passed.")


if __name__ == "__main__":
    main()
