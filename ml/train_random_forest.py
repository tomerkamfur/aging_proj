import argparse
from pathlib import Path

import joblib
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
    parser = argparse.ArgumentParser(description="Train a Random Forest regressor.")
    parser.add_argument(
        "--data",
        default="data_normalized/SI206_2025-05-08/features_stage1_3_early_behavior_death_day.csv",
        help="Path to features CSV.",
    )
    parser.add_argument(
        "--target",
        default="death_day",
        help="Target column name.",
    )
    parser.add_argument("--model-out", required=True, help="Output path for model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees in forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth.")
    parser.add_argument("--k-folds", type=int, default=5, help="K-folds for validation.")
    parser.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Drop feature columns starting with this prefix (repeatable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    drop_cols = set(ID_COLUMNS)
    drop_cols.add(args.target)
    for prefix in args.drop_prefix:
        for col in df.columns:
            if col.startswith(prefix):
                drop_cols.add(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns remain after drops.")

    X = df[feature_cols].astype(float)
    y = df[args.target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
    cv_mae = []
    cv_rmse = []
    cv_r2 = []
    for train_idx, val_idx in kfold.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        cv_mae.append(mean_absolute_error(y_val, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_r2.append(r2_score(y_val, preds))

    print("Validation (CV on train split)")
    print(f"MAE: {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")
    print(f"RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
    print(f"R2: {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    final_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)
    test_preds = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    print("Test (holdout)")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2: {test_r2:.4f}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "target": args.target,
            "metrics": {
                "cv_mae_mean": float(np.mean(cv_mae)),
                "cv_mae_std": float(np.std(cv_mae)),
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_rmse_std": float(np.std(cv_rmse)),
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
            },
        },
        out_path,
    )


if __name__ == "__main__":
    main()
