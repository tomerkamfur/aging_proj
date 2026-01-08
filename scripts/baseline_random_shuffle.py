import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline: sample death_day from training pool."
    )
    parser.add_argument(
        "--data",
        default="data_normalized/SI206_2025-05-08/features_stage1_3_early_behavior_death_day.csv",
        help="Path to features CSV.",
    )
    parser.add_argument("--target", default="death_day", help="Target column name.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    y = df[args.target].astype(float)
    _, y_test, y_train, _ = train_test_split(
        y,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    rng = np.random.default_rng(args.random_state)
    preds = rng.choice(y_train.to_numpy(), size=len(y_test), replace=True)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Train pool size: {len(y_train)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()
