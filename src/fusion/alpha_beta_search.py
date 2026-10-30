

import argparse
import json
from pathlib import Path

import pandas as pd

from reliability import (
    DEFAULT_WINDOW,
    load_detections,
    calculate_modality_reliability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "calibration"
    / "selected_alpha_beta.json"
)

ALPHA_VALUES = [1.0, 1.5, 2.0]
BETA_VALUES = [0.75, 1.0, 1.25]

DARK_WEIGHT = 2.0
WRONG_DIRECTION_PENALTY = 1.0
TIE_TOLERANCE = 0.05


def load_video_pair(rgb_path, ir_path, video_name):
    """Load and validate one RGB-IR detection pair."""

    rgb_all, rgb_valid = load_detections(
        rgb_path,
        f"{video_name} RGB"
    )

    ir_all, ir_valid = load_detections(
        ir_path,
        f"{video_name} IR"
    )

    if rgb_all.empty and ir_all.empty:
        raise ValueError(
            f"Both detection CSVs are empty for {video_name}."
        )

    max_frame = max(
        rgb_all["frame"].max() if not rgb_all.empty else 0,
        ir_all["frame"].max() if not ir_all.empty else 0,
    )

    total_frames = int(max_frame) + 1

    return {
        "name": video_name,
        "rgb_valid": rgb_valid,
        "ir_valid": ir_valid,
        "total_frames": total_frames,
    }


def calculate_pair_means(
    video_data,
    window,
    alpha,
    beta,
):
    """Calculate mean RGB and IR reliability for one video."""

    total_frames = video_data["total_frames"]

    _, _, rgb_reliability = calculate_modality_reliability(
        video_data["rgb_valid"],
        total_frames,
        window,
        alpha,
        beta,
    )

    _, _, ir_reliability = calculate_modality_reliability(
        video_data["ir_valid"],
        total_frames,
        window,
        alpha,
        beta,
    )

    return {
        "rgb_mean": float(rgb_reliability.mean()),
        "ir_mean": float(ir_reliability.mean()),
    }


def evaluate_combination(
    light_data,
    dark_data,
    window,
    alpha,
    beta,
):
    """Evaluate one alpha-beta combination on both videos."""

    light = calculate_pair_means(
        light_data,
        window,
        alpha,
        beta,
    )

    dark = calculate_pair_means(
        dark_data,
        window,
        alpha,
        beta,
    )

    # Correct direction: RGB should dominate in light.
    sep_light = light["rgb_mean"] - light["ir_mean"]

    # Correct direction: IR should dominate in dark.
    sep_dark = dark["ir_mean"] - dark["rgb_mean"]

    # Give the dark environment twice the importance.
    base_score = sep_light + DARK_WEIGHT * sep_dark

    # Apply one penalty if either separation is in the
    # wrong direction.
    wrong_direction = (
        sep_light < 0 or sep_dark < 0
    )

    penalty = (
        WRONG_DIRECTION_PENALTY
        if wrong_direction
        else 0.0
    )

    total_score = base_score - penalty

    return {
        "alpha": alpha,
        "beta": beta,

        "light_rgb_mean": light["rgb_mean"],
        "light_ir_mean": light["ir_mean"],

        "dark_rgb_mean": dark["rgb_mean"],
        "dark_ir_mean": dark["ir_mean"],

        "sep_light": sep_light,
        "sep_dark": sep_dark,

        "base_score": base_score,
        "penalty": penalty,
        "total_score": total_score,

        "wrong_direction": wrong_direction,
    }


def select_best_combination(results):
    """
    Select the highest-scoring combination.

    If candidates are within 0.05 of the maximum score,
    prefer the candidate with the lower alpha.
    """

    highest_score = max(
        result["total_score"]
        for result in results
    )

    near_best = [
        result
        for result in results
        if highest_score - result["total_score"]
        <= TIE_TOLERANCE
    ]

    # Lower alpha is preferred among near-equal candidates.
    # If alpha is also equal, prefer the higher score.
    selected = sorted(
        near_best,
        key=lambda result: (
            result["alpha"],
            -result["total_score"],
            result["beta"],
        ),
    )[0]

    return selected


def save_selected_parameters(
    selected,
    output_path,
    window,
):
    """Save the selected alpha-beta parameters as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "alpha": selected["alpha"],
        "beta": selected["beta"],
        "window_W": window,

        "selection_reason": (
            "Selected using two-video reliability separation: "
            "sep_light + 2 * sep_dark. "
            "Candidates within 0.05 of the highest score "
            "prefer lower alpha."
        ),

        "sep_light": selected["sep_light"],
        "sep_dark": selected["sep_dark"],
        "base_score": selected["base_score"],
        "penalty": selected["penalty"],
        "total_score": selected["total_score"],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    return output_path


def run_search(
    light_rgb,
    light_ir,
    dark_rgb,
    dark_ir,
    window=DEFAULT_WINDOW,
    output_json=OUTPUT_JSON,
):
    """Run the complete nine-combination parameter search."""

    print("\nLoading Light video detection files...")
    light_data = load_video_pair(
        light_rgb,
        light_ir,
        "Light",
    )

    print("Loading Dark video detection files...")
    dark_data = load_video_pair(
        dark_rgb,
        dark_ir,
        "Dark",
    )

    print("\nInput summary:")
    print(
        f"Light frames: {light_data['total_frames']}"
    )
    print(
        f"Dark frames : {dark_data['total_frames']}"
    )

    print(
        f"\nWindow size: {window}"
    )

    results = []

    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:

            result = evaluate_combination(
                light_data,
                dark_data,
                window,
                alpha,
                beta,
            )

            results.append(result)

    selected = select_best_combination(results)

    # Display all nine combinations.
    results_df = pd.DataFrame(results)

    display_columns = [
        "alpha",
        "beta",
        "sep_light",
        "sep_dark",
        "base_score",
        "penalty",
        "total_score",
    ]

    print("\n" + "=" * 105)
    print("DAY 10 - ALPHA-BETA PARAMETER SEARCH")
    print("=" * 105)

    print(
        "\nSeparation definitions:"
        "\n  sep_light = mean RGB reliability - mean IR reliability"
        "\n  sep_dark  = mean IR reliability - mean RGB reliability"
        "\n  total     = sep_light + 2 * sep_dark - penalty"
    )

    print("\nAll nine combinations:")
    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSelected parameters:")
    print(f"Alpha       : {selected['alpha']}")
    print(f"Beta        : {selected['beta']}")
    print(f"Light sep   : {selected['sep_light']:.6f}")
    print(f"Dark sep    : {selected['sep_dark']:.6f}")
    print(f"Total score : {selected['total_score']:.6f}")
    print(f"Penalty     : {selected['penalty']:.6f}")

    output_path = save_selected_parameters(
        selected,
        output_json,
        window,
    )

    print(
        f"\nSelected parameters saved to:\n{output_path}"
    )

    return results_df, selected


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Search alpha-beta parameters using "
            "Light and Dark RGB-IR videos."
        )
    )

    parser.add_argument(
        "--light-rgb",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--light-ir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dark-rgb",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dark-ir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
    )

    args = parser.parse_args()

    run_search(
        light_rgb=args.light_rgb,
        light_ir=args.light_ir,
        dark_rgb=args.dark_rgb,
        dark_ir=args.dark_ir,
        window=args.window,
        output_json=args.output,
    )
