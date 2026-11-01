
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENES = {
    "dark": {
        "rgb": PROJECT_ROOT / "outputs" / "rgb_dark_detection.csv",
        "ir": PROJECT_ROOT / "outputs" / "IR_dark_detection_warped.csv",
        "reliability": PROJECT_ROOT / "outputs" / "reliability_scores_dark_selected.csv",
        "output": PROJECT_ROOT / "outputs" / "fused_detections_adaptive_dark.csv",
    },
    "light": {
        "rgb": PROJECT_ROOT / "outputs" / "rgb_light_detection.csv",
        "ir": PROJECT_ROOT / "outputs" / "IR_light_detection_warped.csv",
        "reliability": PROJECT_ROOT / "outputs" / "reliability_scores_light_selected.csv",
        "output": PROJECT_ROOT / "outputs" / "fused_detections_adaptive_light.csv",
    },
}

PARAMETERS_PATH = (
    PROJECT_ROOT / "outputs" / "calibration" / "selected_alpha_beta.json"
)

MATCH_THRESHOLD = 50.0
UNMATCHED_PENALTY = 0.7
# Scene- and modality-specific gates for unmatched detections.
# RGB-only gates remain at the original 0.15.
# Dark IR is stricter; Light IR is more permissive.
MIN_THRESHOLDS = {
    "dark": {"rgb_only": 0.15, "ir_only": 0.20},
    "light": {"rgb_only": 0.15, "ir_only": 0.08},
}

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

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

DIAGNOSTIC_COLUMNS = [
    "fused_x1",
    "fused_y1",
    "fused_x2",
    "fused_y2",
    "rgb_conf",
    "ir_conf",
    "rgb_reliability",
    "ir_reliability",
    "fusion_type",
    "single_weight",
    "match_distance",
]


def valid_detections(df, name):
    required = {"frame", "x", "y", "width", "height", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="raise")

    if (
        df["frame"].isna().any()
        or not np.isfinite(df["frame"]).all()
        or (df["frame"] < 0).any()
        or not np.equal(df["frame"], np.floor(df["frame"])).all()
    ):
        raise ValueError(f"{name}: frame values must be non-negative integers.")

    for column in ["x", "y", "width", "height", "confidence"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["frame"] = df["frame"].astype(int)

    mask = (
        df[["x", "y", "confidence"]].notna().all(axis=1)
        & np.isfinite(df[["x", "y", "confidence"]]).all(axis=1)
        & (df["confidence"] > 0)
    )

    valid = df.loc[mask].copy()
    valid = valid[
        np.isfinite(valid["width"])
        & np.isfinite(valid["height"])
        & (valid["width"] > 0)
        & (valid["height"] > 0)
    ]

    return valid.reset_index(drop=True)


def load_reliability(path):
    rel = pd.read_csv(path)
    required = {
        "frame", "rgb_reliability", "ir_reliability"
    }
    missing = required - set(rel.columns)
    if missing:
        raise ValueError(f"Reliability CSV missing columns: {sorted(missing)}")

    rel["frame"] = pd.to_numeric(rel["frame"], errors="raise").astype(int)

    for col in ["rgb_reliability", "ir_reliability"]:
        rel[col] = pd.to_numeric(rel[col], errors="raise")

    if rel["frame"].duplicated().any():
        raise ValueError("Reliability CSV contains duplicate frame numbers.")

    values = rel[["rgb_reliability", "ir_reliability"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Reliability values contain NaN or infinity.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Reliability values must be in [0, 1].")

    return rel.set_index("frame").to_dict("index")


def clipped_box(cx, cy, width, height):
    x1 = float(np.clip(cx - width / 2.0, 0, FRAME_WIDTH - 1))
    y1 = float(np.clip(cy - height / 2.0, 0, FRAME_HEIGHT - 1))
    x2 = float(np.clip(cx + width / 2.0, 0, FRAME_WIDTH - 1))
    y2 = float(np.clip(cy + height / 2.0, 0, FRAME_HEIGHT - 1))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid clipped box: ({x1}, {y1}, {x2}, {y2})"
        )

    return x1, y1, x2, y2


def make_row(
    frame,
    fused_x,
    fused_y,
    fused_confidence,
    source,
    rgb_det,
    ir_det,
    rgb_reliability,
    ir_reliability,
    fusion_type,
    single_weight=np.nan,
    match_distance=np.nan,
    box_width=None,
    box_height=None,
):
    if box_width is None or box_height is None:
        raise ValueError("Box dimensions must be provided.")

    x1, y1, x2, y2 = clipped_box(
        fused_x, fused_y, box_width, box_height
    )

    return {
        "frame": int(frame),
        "fused_x": float(fused_x),
        "fused_y": float(fused_y),
        "fused_confidence": float(fused_confidence),
        "source": source,
        "rgb_x": float(rgb_det["x"]) if rgb_det is not None else np.nan,
        "rgb_y": float(rgb_det["y"]) if rgb_det is not None else np.nan,
        "thermal_x": float(ir_det["x"]) if ir_det is not None else np.nan,
        "thermal_y": float(ir_det["y"]) if ir_det is not None else np.nan,
        "fused_x1": x1,
        "fused_y1": y1,
        "fused_x2": x2,
        "fused_y2": y2,
        "rgb_conf": float(rgb_det["confidence"]) if rgb_det is not None else np.nan,
        "ir_conf": float(ir_det["confidence"]) if ir_det is not None else np.nan,
        "rgb_reliability": float(rgb_reliability),
        "ir_reliability": float(ir_reliability),
        "fusion_type": fusion_type,
        "single_weight": float(single_weight),
        "match_distance": float(match_distance),
    }


def greedy_matches(rgb_frame, ir_frame):
    if rgb_frame.empty or ir_frame.empty:
        return [], set(), set()

    rgb_points = rgb_frame[["x", "y"]].to_numpy(dtype=float)
    ir_points = ir_frame[["x", "y"]].to_numpy(dtype=float)

    distances = np.sqrt(
        ((rgb_points[:, None, :] - ir_points[None, :, :]) ** 2).sum(axis=2)
    )

    candidates = sorted(
        (
            (float(distances[i, j]), i, j)
            for i in range(distances.shape[0])
            for j in range(distances.shape[1])
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )

    matched_rgb = set()
    matched_ir = set()
    matches = []

    for distance, i, j in candidates:
        if distance >= MATCH_THRESHOLD:
            break

        if i in matched_rgb or j in matched_ir:
            continue

        matches.append((i, j, distance))
        matched_rgb.add(i)
        matched_ir.add(j)

    return matches, matched_rgb, matched_ir


def run_scene(scene):
    config = SCENES[scene]
    scene_thresholds = MIN_THRESHOLDS[scene]

    for key in ["rgb", "ir", "reliability"]:
        if not config[key].exists():
            raise FileNotFoundError(f"Missing {key} input: {config[key]}")

    if not PARAMETERS_PATH.exists():
        raise FileNotFoundError(f"Missing selected parameters: {PARAMETERS_PATH}")

    with open(PARAMETERS_PATH, "r", encoding="utf-8") as file:
        parameters = json.load(file)

    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    window_w = int(parameters["window_W"])

    rgb = valid_detections(pd.read_csv(config["rgb"]), "RGB")
    ir = valid_detections(pd.read_csv(config["ir"]), "IR")
    reliability = load_reliability(config["reliability"])

    rgb_groups = {
        int(frame): group.reset_index(drop=True)
        for frame, group in rgb.groupby("frame")
    }
    ir_groups = {
        int(frame): group.reset_index(drop=True)
        for frame, group in ir.groupby("frame")
    }

    all_frames = sorted(set(rgb_groups) | set(ir_groups))
    output_rows = []

    single_weight_records = {
        "rgb_only": [],
        "ir_only": [],
    }
    dropped_counts = {
        "rgb_only": 0,
        "ir_only": 0,
    }
    candidate_counts = {
        "rgb_only": 0,
        "ir_only": 0,
    }
    match_count = 0

    for frame in all_frames:
        rgb_frame = rgb_groups.get(frame, pd.DataFrame())
        ir_frame = ir_groups.get(frame, pd.DataFrame())

        rel = reliability.get(frame)
        if rel is None:
            raise ValueError(
                f"Frame {frame} has detections but no reliability row."
            )

        r_rgb = float(rel["rgb_reliability"])
        r_ir = float(rel["ir_reliability"])

        matches, matched_rgb, matched_ir = greedy_matches(
            rgb_frame, ir_frame
        )

        for rgb_index, ir_index, distance in matches:
            rgb_det = rgb_frame.iloc[rgb_index].to_dict()
            ir_det = ir_frame.iloc[ir_index].to_dict()

            w_rgb = r_rgb * float(rgb_det["confidence"])
            w_ir = r_ir * float(ir_det["confidence"])
            w_total = w_rgb + w_ir

            if w_total == 0.0:
                fused_x = (
                    float(rgb_det["x"]) + float(ir_det["x"])
                ) / 2.0
                fused_y = (
                    float(rgb_det["y"]) + float(ir_det["y"])
                ) / 2.0
                fused_confidence = (
                    float(rgb_det["confidence"])
                    + float(ir_det["confidence"])
                ) / 2.0
            else:
                fused_x = (
                    w_rgb * float(rgb_det["x"])
                    + w_ir * float(ir_det["x"])
                ) / w_total
                fused_y = (
                    w_rgb * float(rgb_det["y"])
                    + w_ir * float(ir_det["y"])
                ) / w_total

                denominator = r_rgb + r_ir
                if denominator == 0.0:
                    raise ValueError(
                        "Matched weights are nonzero but reliability sum is zero."
                    )

                fused_confidence = w_total / denominator

            # Day 12 notes specify RGB dimensions for matched boxes.
            output_rows.append(
                make_row(
                    frame=frame,
                    fused_x=fused_x,
                    fused_y=fused_y,
                    fused_confidence=fused_confidence,
                    source="both",
                    rgb_det=rgb_det,
                    ir_det=ir_det,
                    rgb_reliability=r_rgb,
                    ir_reliability=r_ir,
                    fusion_type="matched",
                    match_distance=distance,
                    box_width=float(rgb_det["width"]),
                    box_height=float(rgb_det["height"]),
                )
            )
            match_count += 1

        for index, row in rgb_frame.iterrows():
            if index in matched_rgb:
                continue

            det = row.to_dict()
            single_weight = (
                r_rgb * float(det["confidence"]) * UNMATCHED_PENALTY
            )
            single_weight_records["rgb_only"].append(single_weight)
            candidate_counts["rgb_only"] += 1

            if single_weight <= scene_thresholds["rgb_only"]:
                dropped_counts["rgb_only"] += 1
                continue

            # User-approved override: detector confidence, not single_weight.
            output_rows.append(
                make_row(
                    frame=frame,
                    fused_x=float(det["x"]),
                    fused_y=float(det["y"]),
                    fused_confidence=float(det["confidence"]),
                    source="rgb_only",
                    rgb_det=det,
                    ir_det=None,
                    rgb_reliability=r_rgb,
                    ir_reliability=r_ir,
                    fusion_type="rgb_only",
                    single_weight=single_weight,
                    box_width=float(det["width"]),
                    box_height=float(det["height"]),
                )
            )

        for index, row in ir_frame.iterrows():
            if index in matched_ir:
                continue

            det = row.to_dict()
            single_weight = (
                r_ir * float(det["confidence"]) * UNMATCHED_PENALTY
            )
            single_weight_records["ir_only"].append(single_weight)
            candidate_counts["ir_only"] += 1

            if single_weight <= scene_thresholds["ir_only"]:
                dropped_counts["ir_only"] += 1
                continue

            # The IR CSV supplied here already contains warped RGB-space dimensions.
            # User-approved override: detector confidence, not single_weight.
            output_rows.append(
                make_row(
                    frame=frame,
                    fused_x=float(det["x"]),
                    fused_y=float(det["y"]),
                    fused_confidence=float(det["confidence"]),
                    source="thermal_only",
                    rgb_det=None,
                    ir_det=det,
                    rgb_reliability=r_rgb,
                    ir_reliability=r_ir,
                    fusion_type="ir_only",
                    single_weight=single_weight,
                    box_width=float(det["width"]),
                    box_height=float(det["height"]),
                )
            )

    columns = COMPATIBILITY_COLUMNS + DIAGNOSTIC_COLUMNS
    output = pd.DataFrame(output_rows, columns=columns)

    if not output.empty:
        numeric_columns = [
            "fused_x", "fused_y", "fused_confidence",
            "fused_x1", "fused_y1", "fused_x2", "fused_y2",
        ]
        if not np.isfinite(output[numeric_columns].to_numpy(float)).all():
            raise ValueError("Output contains non-finite fused values.")

        if not output["fused_confidence"].between(0, 1).all():
            raise ValueError("Fused confidence outside [0, 1].")

    config["output"].parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config["output"], index=False)

    # Verify the exact compatibility contract.
    check = pd.read_csv(config["output"])
    if list(check.columns[:9]) != COMPATIBILITY_COLUMNS:
        raise AssertionError("First nine compatibility columns changed.")

    expected_sources = {"both", "rgb_only", "thermal_only"}
    unexpected = set(check["source"].dropna().unique()) - expected_sources
    if unexpected:
        raise AssertionError(f"Unexpected source labels: {unexpected}")

    print("\n" + "=" * 68)
    print(f"DAY 12 - ADAPTIVE FUSION: {scene.upper()}")
    print("=" * 68)
    print(f"Selected alpha, beta, W:       {alpha}, {beta}, {window_w}")
    print(f"Matching threshold:            {MATCH_THRESHOLD:.1f}px")
    print(f"Unmatched penalty:              {UNMATCHED_PENALTY}")
    print(
        "Single-weight gates:             "
        f"RGB-only > {scene_thresholds['rgb_only']:.2f}, "
        f"IR-only > {scene_thresholds['ir_only']:.2f}"
    )
    print(f"Valid RGB detections:           {len(rgb)}")
    print(f"Valid warped IR detections:     {len(ir)}")
    print(f"Matched rows:                   {match_count}")
    print(f"Output rows:                    {len(check)}")
    print("Output source distribution:")
    print(check["source"].value_counts().to_string())

    for modality in ["rgb_only", "ir_only"]:
        values = np.asarray(single_weight_records[modality], dtype=float)
        print(f"\n{modality} gate diagnostics:")
        print(f"  Candidates: {candidate_counts[modality]}")
        print(f"  Dropped:    {dropped_counts[modality]}")

        if candidate_counts[modality]:
            dropped_pct = (
                100.0 * dropped_counts[modality]
                / candidate_counts[modality]
            )
            print(f"  Dropped %:  {dropped_pct:.2f}%")

        if len(values):
            percentiles = np.percentile(
                values, [0, 10, 25, 50, 75, 90, 95, 99, 100]
            )
            print("  single_weight percentiles:")
            for p, value in zip(
                [0, 10, 25, 50, 75, 90, 95, 99, 100],
                percentiles,
            ):
                print(f"    P{p:02d}: {value:.6f}")

    print(f"\nSaved: {config['output']}")
    print("Compatibility schema: PASS")
    print("Confidence range:      PASS" if output.empty or output["fused_confidence"].between(0, 1).all() else "Confidence range: FAIL")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["dark", "light", "both"], default="both")
    args = parser.parse_args()

    scenes = ["dark", "light"] if args.scene == "both" else [args.scene]
    for scene in scenes:
        run_scene(scene)


if __name__ == "__main__":
    main()
