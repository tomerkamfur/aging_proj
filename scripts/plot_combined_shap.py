import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


ID_COLUMNS = {
    "worm_id",
    "cam_id",
    "well_id",
    "recording_id",
    "timestamp",
    "experiment_id",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SHAP plots for Combined multirun models."
    )
    parser.add_argument(
        "--data",
        default="data_normalized/combined_features_days_alive.csv",
        help="Combined features+target CSV path.",
    )
    parser.add_argument(
        "--metrics",
        default="multirun_results/Combined_metrics.csv",
        help="Combined metrics CSV from multirun.",
    )
    parser.add_argument(
        "--target",
        default="days_alive",
        help="Target column name.",
    )
    parser.add_argument(
        "--top-runs",
        type=int,
        default=10,
        help="Number of best runs (by test_r2) to aggregate for SHAP.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split used in multirun.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="RF estimators used in multirun.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="RF max_depth used in multirun.",
    )
    parser.add_argument(
        "--out-dir",
        default="post_multirun_shap",
        help="Output directory for SHAP plots.",
    )
    parser.add_argument(
        "--interaction-feature",
        default="speed_1s_stage5_col1_std_over_mean",
        help="Main feature for SHAP interaction dependence plot.",
    )
    parser.add_argument(
        "--interaction-with",
        default="speed_1s_stage5_col2_std_over_mean",
        help="Feature used for coloring interaction in dependence plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    metrics = pd.read_csv(args.metrics)

    drop_cols = set(ID_COLUMNS)
    drop_cols.add(args.target)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].astype(float)
    y = df[args.target].astype(float)

    top_runs = (
        metrics.sort_values("test_r2", ascending=False).head(args.top_runs)[["run", "seed", "test_r2"]]
    )
    top_runs.to_csv(out_dir / "combined_top_runs_for_shap.csv", index=False)

    shap_blocks = []
    x_blocks = []
    for _, row in top_runs.iterrows():
        seed = int(row["seed"])
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=args.test_size, random_state=seed
        )
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        shap_blocks.append(np.asarray(shap_vals, dtype=float))
        x_blocks.append(X_test.to_numpy(dtype=float))

    shap_all = np.vstack(shap_blocks)
    x_all = np.vstack(x_blocks)

    # Beeswarm summary (feature impact distribution)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_all,
        features=x_all,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    beeswarm_path = out_dir / "combined_shap_summary_beeswarm_top_runs.png"
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Bar summary (mean absolute SHAP)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_all,
        features=x_all,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    bar_path = out_dir / "combined_shap_summary_bar_top_runs.png"
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Save per-feature mean |SHAP| table
    mean_abs_shap = np.abs(shap_all).mean(axis=0)
    mean_abs_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    mean_abs_df.to_csv(out_dir / "combined_mean_abs_shap_top_runs.csv", index=False)

    # Dependence plots for the top 3 features by mean |SHAP|.
    top3_features = mean_abs_df.head(3)["feature"].tolist()
    x_all_df = pd.DataFrame(x_all, columns=feature_cols)
    for rank, feature in enumerate(top3_features, start=1):
        shap.dependence_plot(
            feature,
            shap_all,
            x_all_df,
            show=False,
            interaction_index=None,
        )
        plt.tight_layout()
        safe_feature = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        dep_path = out_dir / f"combined_shap_dependence_{rank:02d}_{safe_feature}.png"
        plt.savefig(dep_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Wrote {dep_path}")

    # Requested explicit interaction dependence plot.
    if args.interaction_feature in feature_cols and args.interaction_with in feature_cols:
        shap.dependence_plot(
            args.interaction_feature,
            shap_all,
            x_all_df,
            interaction_index=args.interaction_with,
            show=False,
        )
        plt.tight_layout()
        main_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.interaction_feature)
        inter_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.interaction_with)
        interaction_path = out_dir / f"combined_shap_interaction_{main_safe}__{inter_safe}.png"
        plt.savefig(interaction_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Wrote {interaction_path}")
    else:
        missing = [
            name
            for name in (args.interaction_feature, args.interaction_with)
            if name not in feature_cols
        ]
        print(f"Skipped interaction plot, missing feature(s): {missing}")

    print(f"Wrote {beeswarm_path}")
    print(f"Wrote {bar_path}")
    print(f"Wrote {out_dir / 'combined_mean_abs_shap_top_runs.csv'}")
    print(f"Wrote {out_dir / 'combined_top_runs_for_shap.csv'}")


if __name__ == "__main__":
    main()
