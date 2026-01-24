import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = [
    ("SI206", "multirun_results/SI206_metrics.csv", "multirun_results/SI206_feature_importance.csv"),
    ("SI216", "multirun_results/SI216_metrics.csv", "multirun_results/SI216_feature_importance.csv"),
    ("Combined", "multirun_results/Combined_metrics.csv", "multirun_results/Combined_feature_importance.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot top-10 feature importance accumulation for best runs."
    )
    parser.add_argument(
        "--out-dir",
        default="multirun_results",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of best runs to include by highest test R2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, metrics_path, imp_path in DATASETS:
        metrics = pd.read_csv(metrics_path)
        top_runs = (
            metrics.sort_values("test_r2", ascending=False)
            .head(args.top_n)["run"]
            .tolist()
        )

        importances = pd.read_csv(imp_path)
        top_imps = importances[importances["run"].isin(top_runs)]
        agg = top_imps.groupby("feature", as_index=False)["importance"].sum()
        agg = agg.sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(agg["feature"], agg["importance"])
        ax.set_title(f"{label}: Sum feature importance (top {args.top_n} runs by test R2)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Summed importance")
        ax.set_xticklabels(agg["feature"], rotation=60, ha="right")
        fig.tight_layout()

        out_path = out_dir / f"top10_feature_importance_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
