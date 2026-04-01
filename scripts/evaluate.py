"""Evaluation metrics for soccer-vision team classification.

Usage:
    python scripts/evaluate.py \
        --predictions output/predictions.csv \
        --ground-truth labels/ground_truth.csv

Ground truth format (produced by scripts/label_tracks.py):
    tracker_id,true_label
    3,team_a
    7,team_b
    12,other

Predictions format (produced by pipeline with save_predictions: true):
    frame_idx,tracker_id,x1,y1,x2,y2,confidence,predicted_label

Metrics reported:
  - Per-detection accuracy   (every row in predictions matched to its track's true label)
  - Per-track accuracy       (majority-vote prediction per track vs true label)
  - Flip rate                (avg label-changes per frame within each track)
  - Confusion matrix         (team_a / team_b / other)
  - Per-class F1             (sklearn classification_report)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_merge(predictions_path: str, ground_truth_path: str) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_path)
    gt_df = pd.read_csv(ground_truth_path)

    required_pred = {"frame_idx", "tracker_id", "predicted_label"}
    required_gt = {"tracker_id", "true_label"}
    _check_columns(pred_df, required_pred, predictions_path)
    _check_columns(gt_df, required_gt, ground_truth_path)

    merged = pred_df.merge(gt_df[["tracker_id", "true_label"]], on="tracker_id", how="inner")
    if len(merged) == 0:
        print("ERROR: No matching tracker_ids between predictions and ground truth.")
        print(f"  predictions tracker_ids (sample): {pred_df['tracker_id'].unique()[:10].tolist()}")
        print(f"  ground truth tracker_ids: {gt_df['tracker_id'].tolist()}")
        sys.exit(1)

    return merged


def _check_columns(df: pd.DataFrame, required: set, path: str) -> None:
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: {path} is missing columns: {missing}")
        sys.exit(1)


def per_detection_accuracy(df: pd.DataFrame) -> float:
    return (df["predicted_label"] == df["true_label"]).mean()


def per_track_accuracy(df: pd.DataFrame) -> float:
    """Per track: majority-vote predicted label vs true label."""
    results = []
    for _, group in df.groupby("tracker_id"):
        true_label = group["true_label"].iloc[0]
        majority_pred = group["predicted_label"].mode()[0]
        results.append(majority_pred == true_label)
    return float(np.mean(results))


def flip_rate(df: pd.DataFrame) -> float:
    """Average fraction of label-changes per frame within each track.

    A perfectly stable track has flip_rate=0. A track that changes label
    every frame has flip_rate=1.
    """
    rates = []
    for _, group in df.groupby("tracker_id"):
        group = group.sort_values("frame_idx")
        labels = group["predicted_label"].tolist()
        if len(labels) < 2:
            continue
        flips = sum(a != b for a, b in zip(labels, labels[1:]))
        rates.append(flips / (len(labels) - 1))
    return float(np.mean(rates)) if rates else 0.0


def confusion_matrix_str(df: pd.DataFrame, labels: list) -> str:
    """Return a formatted confusion matrix string."""
    from collections import defaultdict

    counts: dict = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        counts[row["true_label"]][row["predicted_label"]] += 1

    col_w = 12
    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    lines = [header]
    for true_l in labels:
        row_str = f"{true_l:>{col_w}}"
        for pred_l in labels:
            row_str += f"{counts[true_l][pred_l]:>{col_w}}"
        lines.append(row_str)
    return "\n".join(lines)


def classification_report_str(df: pd.DataFrame) -> str:
    from sklearn.metrics import classification_report

    return classification_report(
        df["true_label"],
        df["predicted_label"],
        labels=["team_a", "team_b", "other"],
        zero_division=0,
    )


# ---------------------------------------------------------------------------
# Benchmark targets
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "per_detection_accuracy": [("Acceptable", 0.80), ("Good", 0.90), ("Production", 0.95)],
    "per_track_accuracy":     [("Acceptable", 0.85), ("Good", 0.95), ("Production", 0.98)],
    "flip_rate":              [("Acceptable", 0.05), ("Good", 0.02), ("Production", 0.005)],
}


def _grade(metric: str, value: float) -> str:
    tiers = BENCHMARKS[metric]
    if metric == "flip_rate":
        # Lower is better
        for label, threshold in reversed(tiers):
            if value <= threshold:
                return label
        return "Below acceptable"
    else:
        # Higher is better
        for label, threshold in reversed(tiers):
            if value >= threshold:
                return label
        return "Below acceptable"


def print_report(
    df: pd.DataFrame,
    predictions_path: str,
    ground_truth_path: str,
) -> None:
    labels = ["team_a", "team_b", "other"]

    det_acc = per_detection_accuracy(df)
    trk_acc = per_track_accuracy(df)
    fr = flip_rate(df)

    n_tracks = df["tracker_id"].nunique()
    n_detections = len(df)

    print()
    print("=" * 60)
    print("  SOCCER VISION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Predictions :  {predictions_path}")
    print(f"  Ground truth:  {ground_truth_path}")
    print(f"  Tracks labeled: {n_tracks}")
    print(f"  Detection rows: {n_detections}")
    print()

    print("─" * 60)
    print("  CORE METRICS")
    print("─" * 60)

    metric_rows = [
        ("Per-detection accuracy", det_acc, "per_detection_accuracy", ">"),
        ("Per-track accuracy",     trk_acc, "per_track_accuracy",     ">"),
        ("Flip rate",              fr,       "flip_rate",              "<"),
    ]
    for name, value, key, direction in metric_rows:
        grade = _grade(key, value)
        bar = _ascii_bar(value if direction == ">" else 1 - value)
        print(f"  {name:<28} {value:>6.1%}  [{bar}]  {grade}")

    print()
    print("  Benchmark targets:")
    print(f"  {'Metric':<28} {'Acceptable':>12} {'Good':>8} {'Production':>12}")
    print(f"  {'':<28} {'> 80%':>12} {'> 90%':>8} {'> 95%':>12}  per-detection")
    print(f"  {'':<28} {'> 85%':>12} {'> 95%':>8} {'> 98%':>12}  per-track")
    print(f"  {'':<28} {'< 5%':>12} {'< 2%':>8} {'< 0.5%':>12}  flip rate")
    print()

    print("─" * 60)
    print("  CONFUSION MATRIX (rows=true, cols=predicted)")
    print("─" * 60)
    print(confusion_matrix_str(df, labels))
    print()

    print("─" * 60)
    print("  PER-CLASS F1 (sklearn classification_report)")
    print("─" * 60)
    print(classification_report_str(df))
    print("=" * 60)


def _ascii_bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return "#" * filled + "-" * (width - filled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate soccer-vision team classification predictions."
    )
    parser.add_argument(
        "--predictions",
        default="output/predictions.csv",
        help="Path to predictions.csv (default: output/predictions.csv)",
    )
    parser.add_argument(
        "--ground-truth",
        default="labels/ground_truth.csv",
        help="Path to ground_truth.csv (default: labels/ground_truth.csv)",
    )
    args = parser.parse_args()

    for path in [args.predictions, args.ground_truth]:
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    df = load_and_merge(args.predictions, args.ground_truth)
    print_report(df, args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()
