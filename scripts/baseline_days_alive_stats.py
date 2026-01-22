import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baselines for days_alive: random normal and random shuffle."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="One or more targets_days_alive.csv paths.",
    )
    parser.add_argument("--label", required=True, help="Label for this run.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        default="ml/analysis/baseline_days_alive_stats.csv",
        help="Output CSV for results.",
    )
    return parser.parse_args()


def eval_baselines(y, test_size, random_state):
    _, y_test, y_train, _ = train_test_split(
        y,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)

    # Random normal baseline
    mean = float(y_train.mean())
    std = float(y_train.std(ddof=0))
    preds_normal = rng.normal(loc=mean, scale=std, size=len(y_test))
    normal = {
        "train_mean": mean,
        "train_std": std,
        "mae": float(mean_absolute_error(y_test, preds_normal)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds_normal))),
        "r2": float(r2_score(y_test, preds_normal)),
    }

    # Random shuffle baseline
    preds_shuffle = rng.choice(y_train.to_numpy(), size=len(y_test), replace=True)
    shuffle = {
        "train_pool_size": int(len(y_train)),
        "mae": float(mean_absolute_error(y_test, preds_shuffle)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds_shuffle))),
        "r2": float(r2_score(y_test, preds_shuffle)),
    }

    return normal, shuffle


def main():
    args = parse_args()
    targets = []
    for t in args.targets:
        path = Path(t)
        if not path.exists():
            raise FileNotFoundError(f"Missing target file: {path}")
        df = pd.read_csv(path)
        if "days_alive" not in df.columns:
            raise ValueError(f"days_alive column not found in {path}")
        targets.append(df["days_alive"].astype(float))

    y = pd.concat(targets, ignore_index=True)
    normal, shuffle = eval_baselines(y, args.test_size, args.random_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(
        {
            "label": args.label,
            "baseline": "random_normal",
            "train_mean": normal["train_mean"],
            "train_std": normal["train_std"],
            "train_pool_size": "",
            "mae": normal["mae"],
            "rmse": normal["rmse"],
            "r2": normal["r2"],
        }
    )
    rows.append(
        {
            "label": args.label,
            "baseline": "random_shuffle",
            "train_mean": "",
            "train_std": "",
            "train_pool_size": shuffle["train_pool_size"],
            "mae": shuffle["mae"],
            "rmse": shuffle["rmse"],
            "r2": shuffle["r2"],
        }
    )

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        cols = [
            "label",
            "baseline",
            "train_mean",
            "train_std",
            "train_pool_size",
            "mae",
            "rmse",
            "r2",
        ]
        writer = pd.DataFrame(rows)
        if write_header:
            writer.to_csv(f, index=False)
        else:
            writer.to_csv(f, index=False, header=False)

    print(f"Wrote {out_path}")
    for row in rows:
        print(
            f"{row['label']} {row['baseline']}: MAE {row['mae']:.4f}, "
            f"RMSE {row['rmse']:.4f}, R2 {row['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
