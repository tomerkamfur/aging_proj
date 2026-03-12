import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASETS = [
    ("SI216", "multirun_results/SI216_metrics.csv"),
    ("SI206", "multirun_results/SI206_metrics.csv"),
    ("Combined", "multirun_results/Combined_metrics.csv"),
    ("SI216_train_shuffled_test_regular", "multirun_results/SI216_train_shuffled_test_regular_metrics.csv"),
    ("SI206_train_shuffled_test_regular", "multirun_results/SI206_train_shuffled_test_regular_metrics.csv"),
    ("Combined_train_shuffled_test_regular", "multirun_results/Combined_train_shuffled_test_regular_metrics.csv"),
    ("SI216_shuffled", "multirun_results/SI216_shuffled_metrics.csv"),
    ("SI206_shuffled", "multirun_results/SI206_shuffled_metrics.csv"),
    ("Combined_shuffled", "multirun_results/Combined_shuffled_metrics.csv"),
]

METRICS = ["test_mae", "test_rmse", "test_r2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot beeswarm graphs for multirun metrics.")
    parser.add_argument(
        "--out-dir",
        default="multirun_results",
        help="Output directory for plots.",
    )
    return parser.parse_args()


def load_metrics():
    rows = []
    for label, path_str in DATASETS:
        path = Path(path_str)
        df = pd.read_csv(path)
        df["dataset"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_metric(df, metric, out_dir: Path):
    order = [label for label, _ in DATASETS]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.swarmplot(
        data=df,
        x="dataset",
        y=metric,
        order=order,
        size=2.5,
        ax=ax,
    )

    # Overlay mean and median markers.
    for idx, label in enumerate(order):
        values = df.loc[df["dataset"] == label, metric].dropna()
        if values.empty:
            continue
        mean_val = values.mean()
        median_val = values.median()
        ax.scatter(idx, mean_val, color="purple", marker="D", s=35, zorder=5, label=None)
        ax.scatter(idx, median_val, color="red", marker="D", s=35, zorder=5, label=None)

    ax.set_title(f"Beeswarm of {metric}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.text(
        0.99,
        0.02,
        "purple diamond = mean\nred diamond = median",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    out_path = out_dir / f"beeswarm_{metric}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics()

    for metric in METRICS:
        plot_metric(df, metric, out_dir)


if __name__ == "__main__":
    main()
