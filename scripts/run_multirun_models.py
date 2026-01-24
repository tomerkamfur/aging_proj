import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


ID_COLUMNS = {
    "worm_id",
    "cam_id",
    "well_id",
    "recording_id",
    "timestamp",
    "experiment_id",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple RF trainings and log metrics.")
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of runs per dataset."
    )
    parser.add_argument("--k-folds", type=int, default=5, help="K-fold CV on train.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees in forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs.")
    parser.add_argument(
        "--out-dir",
        default="multirun_results",
        help="Output folder for metrics and importances.",
    )
    return parser.parse_args()


def load_dataset(path: Path, target: str):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column not found in {path}")
    drop_cols = set(ID_COLUMNS)
    drop_cols.add(target)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].astype(float)
    y = df[target].astype(float)
    return X, y, feature_cols


def run_once(X, y, feature_cols, args, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=seed
    )

    cv_mae = []
    cv_rmse = []
    cv_r2 = []
    if args.k_folds > 1:
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
        for train_idx, val_idx in kfold.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=seed,
                n_jobs=args.n_jobs,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_mae.append(mean_absolute_error(y_val, preds))
            cv_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
            cv_r2.append(r2_score(y_val, preds))

    final_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    final_model.fit(X_train, y_train)
    test_preds = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    metrics = {
        "cv_mae_mean": float(np.mean(cv_mae)) if cv_mae else "",
        "cv_mae_std": float(np.std(cv_mae)) if cv_mae else "",
        "cv_rmse_mean": float(np.mean(cv_rmse)) if cv_rmse else "",
        "cv_rmse_std": float(np.std(cv_rmse)) if cv_rmse else "",
        "cv_r2_mean": float(np.mean(cv_r2)) if cv_r2 else "",
        "cv_r2_std": float(np.std(cv_r2)) if cv_r2 else "",
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": final_model.feature_importances_,
        }
    )
    return metrics, importances


def write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "SI206": "data_normalized/SI206_2025-05-08/features_days_alive.csv",
        "SI216": "data_normalized/SI216_2025-04-12/features_days_alive.csv",
        "Combined": "data_normalized/combined_features_days_alive.csv",
        "SI206_shuffled": "data_normalized/SI206_2025-05-08/features_days_alive_shuffled.csv",
        "SI216_shuffled": "data_normalized/SI216_2025-04-12/features_days_alive_shuffled.csv",
        "Combined_shuffled": "data_normalized/combined_features_days_alive_shuffled.csv",
    }

    for name, path_str in datasets.items():
        path = Path(path_str)
        X, y, feature_cols = load_dataset(path, target="days_alive")

        metrics_rows = []
        import_rows = []
        for run_idx in range(args.runs):
            seed = 42 + run_idx
            metrics, importances = run_once(X, y, feature_cols, args, seed)
            metrics_row = {
                "dataset": name,
                "run": run_idx + 1,
                "seed": seed,
                **metrics,
            }
            metrics_rows.append(metrics_row)

            importances = importances.copy()
            importances["dataset"] = name
            importances["run"] = run_idx + 1
            importances["seed"] = seed
            import_rows.append(importances)

        metrics_path = out_dir / f"{name}_metrics.csv"
        import_path = out_dir / f"{name}_feature_importance.csv"
        write_metrics(metrics_path, metrics_rows)
        pd.concat(import_rows, ignore_index=True).to_csv(import_path, index=False)
        print(f"Wrote {metrics_path} and {import_path}")


if __name__ == "__main__":
    main()
