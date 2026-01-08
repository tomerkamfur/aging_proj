import csv
from pathlib import Path

import pandas as pd
from scipy import stats


EXPERIMENT_ID = "SI206_2025-05-08"
BASE_DIR = Path("data_normalized") / EXPERIMENT_ID
INPUT_PATH = BASE_DIR / "features_stage1_3_early_behavior_death_day.csv"
OUTPUT_PATH = Path("ml") / "analysis" / "stage_duration_correlations.csv"

FEATURES = [
    "stage_1_frames",
    "stage_2_frames",
    "stage_3_frames",
    "stage_1_2_frames",
    "stage_1_3_frames",
]


def main():
    df = pd.read_csv(INPUT_PATH)
    if "death_day" not in df.columns:
        raise ValueError("death_day column not found in input file.")

    rows = []
    for feature in FEATURES:
        if feature not in df.columns:
            continue
        x = df[feature].astype(float)
        y = df["death_day"].astype(float)

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        lin = stats.linregress(x, y)

        rows.append(
            {
                "feature": feature,
                "n": int(len(df)),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "linreg_slope": float(lin.slope),
                "linreg_intercept": float(lin.intercept),
                "linreg_p": float(lin.pvalue),
                "linreg_r2": float(lin.rvalue ** 2),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for row in rows:
        print(
            f"{row['feature']}: pearson r={row['pearson_r']:.3f} (p={row['pearson_p']:.3g}), "
            f"spearman r={row['spearman_r']:.3f} (p={row['spearman_p']:.3g})"
        )


if __name__ == "__main__":
    main()
