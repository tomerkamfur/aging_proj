import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SI206_EXPERIMENT_ID = "SI206_2025-05-08"
SI216_EXPERIMENT_ID = "SI216_2025-04-12"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot feature value vs days_alive for top Combined multirun features, "
            "colored by experiment."
        )
    )
    parser.add_argument(
        "--metrics",
        default="multirun_results/Combined_metrics.csv",
        help="Combined metrics CSV from multirun.",
    )
    parser.add_argument(
        "--importances",
        default="multirun_results/Combined_feature_importance.csv",
        help="Combined feature importances CSV from multirun.",
    )
    parser.add_argument(
        "--data",
        default="data_normalized/combined_features_days_alive.csv",
        help="Combined features+target CSV.",
    )
    parser.add_argument(
        "--top-runs",
        type=int,
        default=10,
        help="How many best runs (by test_r2) to use for feature aggregation.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
        help="How many top features to plot.",
    )
    parser.add_argument(
        "--out-dir",
        default="post_multirun_correlation",
        help="Output folder for scatter plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics)
    importances_path = Path(args.importances)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    importances = pd.read_csv(importances_path)
    data = pd.read_csv(data_path)

    top_runs = (
        metrics.sort_values("test_r2", ascending=False).head(args.top_runs)["run"].tolist()
    )
    agg = (
        importances[importances["run"].isin(top_runs)]
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .head(args.top_features)
    )
    top_features = agg["feature"].tolist()

    si206 = data[data["experiment_id"] == SI206_EXPERIMENT_ID]
    si216 = data[data["experiment_id"] == SI216_EXPERIMENT_ID]

    for rank, feature in enumerate(top_features, start=1):
        if feature not in data.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            si206[feature].astype(float),
            si206["days_alive"].astype(float),
            color="purple",
            alpha=0.8,
            s=22,
            label="SI206",
            edgecolor="none",
        )
        ax.scatter(
            si216[feature].astype(float),
            si216["days_alive"].astype(float),
            color="green",
            alpha=0.8,
            s=22,
            label="SI216",
            edgecolor="none",
        )

        # Add linear trend lines for each dataset and the combined data.
        x_all = data[feature].astype(float)
        y_all = data["days_alive"].astype(float)
        x_206 = si206[feature].astype(float)
        y_206 = si206["days_alive"].astype(float)
        x_216 = si216[feature].astype(float)
        y_216 = si216["days_alive"].astype(float)

        def add_trend_line(x: pd.Series, y: pd.Series, color: str, label: str):
            if len(x) < 2:
                return
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()], dtype=float)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color=color, linewidth=2.0, linestyle="--", label=label)

        add_trend_line(x_all, y_all, "black", "Combined trend")
        add_trend_line(x_206, y_206, "purple", "SI206 trend")
        add_trend_line(x_216, y_216, "green", "SI216 trend")

        ax.set_xlabel(feature)
        ax.set_ylabel("days_alive")
        ax.set_title(f"Combined top feature #{rank}: {feature}")
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = out_dir / f"{rank:02d}_{feature}_vs_days_alive.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
