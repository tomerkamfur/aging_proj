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
        "--train-data",
        default="",
        help="Optional training features CSV (overrides --data for training).",
    )
    parser.add_argument(
        "--test-data",
        default="",
        help="Optional test features CSV (overrides --data for testing).",
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
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for training.")
    parser.add_argument(
        "--importance-out",
        default="ml/models/rf_death_day_feature_importance.csv",
        help="Output CSV for feature importances.",
    )
    parser.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Drop feature columns starting with this prefix (repeatable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train_data) if args.train_data else Path(args.data)
    test_path = Path(args.test_data) if args.test_data else None

    if not train_path.exists():
        raise FileNotFoundError(f"Missing data file: {train_path}")

    df = pd.read_csv(train_path)
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

    if test_path:
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test data file: {test_path}")
        df_test = pd.read_csv(test_path)
        if args.target not in df_test.columns:
            raise ValueError(f"Target column not found in test file: {args.target}")
        X_train, y_train = X, y
        drop_cols_test = set(ID_COLUMNS)
        drop_cols_test.add(args.target)
        for prefix in args.drop_prefix:
            for col in df_test.columns:
                if col.startswith(prefix):
                    drop_cols_test.add(col)
        feature_cols_test = [c for c in df_test.columns if c not in drop_cols_test]
        if feature_cols_test != feature_cols:
            raise ValueError("Train/test feature columns do not match.")
        X_test = df_test[feature_cols].astype(float)
        y_test = df_test[args.target].astype(float)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    cv_mae = []
    cv_rmse = []
    cv_r2 = []
    if args.k_folds > 1:
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
        for train_idx, val_idx in kfold.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
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
        n_jobs=args.n_jobs,
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
    metrics = {
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }
    if cv_mae:
        metrics.update(
            {
                "cv_mae_mean": float(np.mean(cv_mae)),
                "cv_mae_std": float(np.std(cv_mae)),
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_rmse_std": float(np.std(cv_rmse)),
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
            }
        )

    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "target": args.target,
            "metrics": metrics,
        },
        out_path,
    )

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(args.importance_out, index=False)


if __name__ == "__main__":
    main()
