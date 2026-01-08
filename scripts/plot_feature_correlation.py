import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot feature vs death_day with trend line and stats."
    )
    parser.add_argument(
        "--data",
        default="data_normalized/SI206_2025-05-08/features_stage1_3_early_behavior_death_day.csv",
        help="Path to features CSV.",
    )
    parser.add_argument("--feature", default="stage_1_frames", help="Feature name.")
    parser.add_argument("--target", default="death_day", help="Target column name.")
    parser.add_argument(
        "--exclude-worm-id",
        action="append",
        default=[],
        help="Worm ID to exclude (repeatable).",
    )
    parser.add_argument(
        "--out",
        default="ml/analysis/stage_1_frames_vs_death_day.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)
    if args.feature not in df.columns:
        raise ValueError(f"Feature not found: {args.feature}")
    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    if args.exclude_worm_id:
        df = df[~df["worm_id"].isin(args.exclude_worm_id)]

    x = df[args.feature].astype(float)
    y = df[args.target].astype(float)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    lin = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.7, edgecolor="none")

    xs = pd.Series([x.min(), x.max()])
    ys = lin.intercept + lin.slope * xs
    ax.plot(xs, ys, color="black", linewidth=2, label="Linear trend")

    ax.set_title(f"{args.feature} vs {args.target}")
    ax.set_xlabel(args.feature)
    ax.set_ylabel(args.target)
    ax.legend(loc="best")

    stats_text = (
        f"n={len(df)}\n"
        f"pearson r={pearson_r:.3f} (p={pearson_p:.3g})\n"
        f"spearman r={spearman_r:.3f} (p={spearman_p:.3g})\n"
        f"slope={lin.slope:.6f}, intercept={lin.intercept:.3f}\n"
        f"r2={lin.rvalue ** 2:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
