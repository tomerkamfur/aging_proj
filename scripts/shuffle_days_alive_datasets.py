import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle days_alive targets in datasets.")
    parser.add_argument(
        "--si206",
        default="data_normalized/SI206_2025-05-08/features_days_alive.csv",
        help="SI206 features_days_alive.csv path.",
    )
    parser.add_argument(
        "--si216",
        default="data_normalized/SI216_2025-04-12/features_days_alive.csv",
        help="SI216 features_days_alive.csv path.",
    )
    parser.add_argument(
        "--combined",
        default="data_normalized/combined_features_days_alive.csv",
        help="Combined features_days_alive.csv path.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_normalized",
        help="Output directory for shuffled datasets.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def shuffle_days_alive(path: Path, out_path: Path, rng: np.random.Generator):
    df = pd.read_csv(path)
    if "days_alive" not in df.columns:
        raise ValueError(f"days_alive column not found in {path}")
    shuffled = df["days_alive"].to_numpy().copy()
    rng.shuffle(shuffled)
    df["days_alive"] = shuffled
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    out_dir = Path(args.out_dir)
    shuffle_days_alive(
        Path(args.si206),
        out_dir / "SI206_2025-05-08" / "features_days_alive_shuffled.csv",
        rng,
    )
    shuffle_days_alive(
        Path(args.si216),
        out_dir / "SI216_2025-04-12" / "features_days_alive_shuffled.csv",
        rng,
    )
    shuffle_days_alive(
        Path(args.combined),
        out_dir / "combined_features_days_alive_shuffled.csv",
        rng,
    )
    print("Wrote shuffled datasets.")


if __name__ == "__main__":
    main()
