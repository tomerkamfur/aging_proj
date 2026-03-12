# PCA Multirun Results

This folder contains 100-run Random Forest comparisons between two feature spaces on three datasets:
- `regular` (original engineered features)
- `pca10` (top 10 principal components from standardized features)

## Outputs
- Metrics CSVs: `*_regular_vs_pca10_metrics.csv`
- PCA loadings: `pca_loadings/*_top10_loadings.csv` (what each PC contains)
- Saved models: `saved_models/*.joblib` (best/mean/worst by test R2 per dataset+feature_space)
- Beeswarm plots:
  - `beeswarm_pca_vs_regular_test_mae.png`
  - `beeswarm_pca_vs_regular_test_rmse.png`
  - `beeswarm_pca_vs_regular_test_r2.png`

## Aggregated Metrics (100 runs each)

| Dataset | Feature Space | Runs | Mean R2 | Std R2 | Best R2 | Worst R2 | Mean MAE | Mean RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Combined | pca10 | 100 | -0.2794 | 0.2099 | 0.0945 | -1.0088 | 2.7477 | 3.4187 |
| Combined | regular | 100 | -0.1507 | 0.1975 | 0.1751 | -0.7605 | 2.5794 | 3.2465 |
| SI206 | pca10 | 100 | -0.3942 | 0.3041 | 0.1610 | -1.3425 | 2.7805 | 3.6786 |
| SI206 | regular | 100 | -0.2921 | 0.3113 | 0.2141 | -1.6047 | 2.7034 | 3.5369 |
| SI216 | pca10 | 100 | -0.3554 | 0.7818 | 0.4030 | -4.9425 | 2.5626 | 3.0377 |
| SI216 | regular | 100 | -0.3215 | 0.9478 | 0.4197 | -5.6439 | 2.5110 | 2.9599 |

## Best Model Metrics (By Highest Test R2)

| Dataset | Feature Space | Run | Seed | Best R2 | MAE (best model) | RMSE (best model) |
|---|---:|---:|---:|---:|---:|---:|
| Combined | pca10 | 30 | 71 | 0.0945 | 2.3400 | 2.9015 |
| Combined | regular | 90 | 131 | 0.1751 | 2.1343 | 2.6360 |
| SI206 | pca10 | 85 | 126 | 0.1610 | 2.2390 | 2.8862 |
| SI206 | regular | 51 | 92 | 0.2141 | 1.1072 | 1.2611 |
| SI216 | pca10 | 93 | 134 | 0.4030 | 1.8245 | 2.0647 |
| SI216 | regular | 12 | 53 | 0.4197 | 1.3458 | 1.4625 |

## Statistical Comparison: Regular vs PCA10

Paired Wilcoxon signed-rank tests were computed across the 100 matched runs for each dataset. These p-values are based on the full run distributions, not on the single best model. Deltas are defined as `regular - pca10`, so positive delta favors `regular` for `R2`, while negative delta favors `regular` for `MAE` and `RMSE`.

P-value calculation details:
- Runs were matched by `run` / `seed` across the two feature spaces.
- For each metric, a paired delta was computed for every matched run: `delta = regular - pca10`.
- A two-sided Wilcoxon signed-rank test was then applied to the 100 paired deltas.
- Reported effect sizes are the mean and median of those paired deltas.

| Dataset | Metric | p-value | Mean delta (regular - pca10) | Median delta (regular - pca10) |
|---|---|---:|---:|---:|
| SI206 | R2 | 0.000885 | 0.1020 | 0.1175 |
| SI206 | MAE | 0.032187 | -0.0770 | -0.0368 |
| SI206 | RMSE | 0.000529 | -0.1418 | -0.1350 |
| SI216 | R2 | 0.006741 | 0.0339 | 0.0750 |
| SI216 | MAE | 0.074344 | -0.0516 | -0.0668 |
| SI216 | RMSE | 0.002655 | -0.0779 | -0.1121 |
| Combined | R2 | 0.000000000150 | 0.1287 | 0.1170 |
| Combined | MAE | 0.000000000006 | -0.1683 | -0.1638 |
| Combined | RMSE | 0.000000000122 | -0.1722 | -0.1618 |

Interpretation:
- For `SI206`, `regular` is significantly better than `pca10` on all three metrics. The effect is strongest on `R2` and `RMSE`, and smaller but still significant on `MAE`.
- For `SI216`, `regular` is significantly better than `pca10` on `R2` and `RMSE`, but the `MAE` difference is weaker (`p = 0.074344`) and does not meet a conventional 0.05 threshold.
- For `Combined`, the evidence is strong across all three metrics that `regular` outperforms `pca10`. The positive `R2` delta and negative `MAE`/`RMSE` deltas are all large and highly significant.
- Overall, the PCA compression to 10 components appears to lose predictive signal relative to the original feature space, especially for the combined dataset.

## Saved Model Success Rate

- Expected saved models: **18**
- Present on disk: **18** (100.0%)
- Loadable with `joblib.load`: **18** (100.0%)
