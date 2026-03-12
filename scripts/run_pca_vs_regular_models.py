import argparse
import csv
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")


ID_COLUMNS = {
    "worm_id",
    "cam_id",
    "well_id",
    "recording_id",
    "timestamp",
    "experiment_id",
}

DATASETS = {
    "SI206": "data_normalized/SI206_2025-05-08/features_days_alive.csv",
    "SI216": "data_normalized/SI216_2025-04-12/features_days_alive.csv",
    "Combined": "data_normalized/combined_features_days_alive.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare regular features vs top-10 PCA features across 100 runs."
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees in forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for RF.")
    parser.add_argument("--top-pcs", type=int, default=10, help="Number of PCA components to keep.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training and only generate beeswarm plots from existing metrics CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default="pca_multirun_results",
        help="Output folder for metrics/loadings/models.",
    )
    return parser.parse_args()


def load_dataset(path: Path, target: str = "days_alive"):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column not found in {path}")
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS and c != target]
    X = df[feature_cols].astype(float)
    y = df[target].astype(float)
    return X, y, feature_cols


def save_pca_loadings(X: pd.DataFrame, feature_cols, out_path: Path, top_pcs: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(top_pcs, X.shape[1]))
    pca.fit(X_scaled)

    rows = []
    for pc_idx in range(pca.n_components_):
        pc_name = f"PC{pc_idx + 1}"
        evr = float(pca.explained_variance_ratio_[pc_idx])
        for feat, loading in zip(feature_cols, pca.components_[pc_idx]):
            rows.append(
                {
                    "pc": pc_name,
                    "explained_variance_ratio": evr,
                    "feature": feat,
                    "loading": float(loading),
                    "abs_loading": float(abs(loading)),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def fit_eval_regular(X_train, X_test, y_train, y_test, args, seed):
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "test_r2": float(r2_score(y_test, preds)),
    }
    return model, metrics


def fit_eval_pca(X_train, X_test, y_train, y_test, args, seed):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_comp = min(args.top_pcs, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train_pca, y_train)
    preds = model.predict(X_test_pca)
    metrics = {
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "test_r2": float(r2_score(y_test, preds)),
    }
    return model, scaler, pca, metrics


def write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_best_mean_worst(metrics_df: pd.DataFrame):
    best_idx = metrics_df["test_r2"].idxmax()
    worst_idx = metrics_df["test_r2"].idxmin()
    mean_r2 = metrics_df["test_r2"].mean()
    mean_idx = (metrics_df["test_r2"] - mean_r2).abs().idxmin()
    return {
        "best": int(metrics_df.loc[best_idx, "run"]),
        "mean": int(metrics_df.loc[mean_idx, "run"]),
        "worst": int(metrics_df.loc[worst_idx, "run"]),
    }


def train_for_run(X, y, args, seed, space: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=seed
    )
    if space == "regular":
        model, metrics = fit_eval_regular(X_train, X_test, y_train, y_test, args, seed)
        return {
            "model": model,
            "metrics": metrics,
            "scaler": None,
            "pca": None,
        }
    model, scaler, pca, metrics = fit_eval_pca(X_train, X_test, y_train, y_test, args, seed)
    return {
        "model": model,
        "metrics": metrics,
        "scaler": scaler,
        "pca": pca,
    }


def plot_pca_vs_regular_beeswarms(all_rows, out_dir: Path):
    df = pd.DataFrame(all_rows)
    df["dataset_space"] = df["dataset"] + "_" + df["feature_space"]
    order = [
        "SI206_regular",
        "SI206_pca10",
        "SI216_regular",
        "SI216_pca10",
        "Combined_regular",
        "Combined_pca10",
    ]
    metrics = ["test_mae", "test_rmse", "test_r2"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.swarmplot(
            data=df,
            x="dataset_space",
            y=metric,
            order=order,
            size=2.5,
            ax=ax,
        )
        ax.set_title(f"PCA10 vs Regular Beeswarm: {metric}")
        ax.set_xlabel("Dataset + Feature Space")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right")

        out_path = out_dir / f"beeswarm_pca_vs_regular_{metric}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "saved_models"
    loadings_dir = out_dir / "pca_loadings"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    loadings_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = []
        for dataset_name in DATASETS.keys():
            metrics_path = out_dir / f"{dataset_name}_regular_vs_pca10_metrics.csv"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file for plot-only mode: {metrics_path}")
            rows.extend(pd.read_csv(metrics_path).to_dict("records"))
        plot_pca_vs_regular_beeswarms(rows, out_dir)
        return

    global_rows = []
    for dataset_name, path_str in DATASETS.items():
        path = Path(path_str)
        X, y, feature_cols = load_dataset(path)

        # Save interpretable PCA content (loadings) once per dataset.
        save_pca_loadings(
            X,
            feature_cols,
            loadings_dir / f"{dataset_name}_top{args.top_pcs}_loadings.csv",
            args.top_pcs,
        )

        all_rows = []
        for run_idx in range(args.runs):
            run_num = run_idx + 1
            seed = 42 + run_idx

            reg = train_for_run(X, y, args, seed, "regular")
            pca = train_for_run(X, y, args, seed, "pca10")

            all_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_space": "regular",
                    "run": run_num,
                    "seed": seed,
                    **reg["metrics"],
                }
            )
            global_rows.extend(all_rows[-2:])
            all_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_space": "pca10",
                    "run": run_num,
                    "seed": seed,
                    **pca["metrics"],
                }
            )

        metrics_path = out_dir / f"{dataset_name}_regular_vs_pca10_metrics.csv"
        write_metrics(metrics_path, all_rows)
        print(f"Wrote {metrics_path}")

        metrics_df = pd.DataFrame(all_rows)
        for space in ("regular", "pca10"):
            sub = metrics_df[metrics_df["feature_space"] == space].reset_index(drop=True)
            picks = select_best_mean_worst(sub)
            for tag, run_num in picks.items():
                seed = 42 + (run_num - 1)
                artifact = train_for_run(X, y, args, seed, space)
                payload = {
                    "dataset": dataset_name,
                    "feature_space": space,
                    "selection": tag,
                    "run": run_num,
                    "seed": seed,
                    "model": artifact["model"],
                    "metrics": artifact["metrics"],
                    "feature_cols": feature_cols,
                    "scaler": artifact["scaler"],
                    "pca": artifact["pca"],
                    "top_pcs": args.top_pcs,
                }
                out_model = models_dir / f"{dataset_name}_{space}_{tag}_model.joblib"
                joblib.dump(payload, out_model)
                print(f"Wrote {out_model}")

    plot_pca_vs_regular_beeswarms(global_rows, out_dir)


if __name__ == "__main__":
    main()
