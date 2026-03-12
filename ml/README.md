ML workspace

This folder contains model training code and artifacts for the aging project.

Getting started
- Input features: to be regenerated after corrected data is provided.

Baseline model
- `train_random_forest.py` trains a Random Forest regressor.
- You must specify the target column name from the feature table.

Example
python ml/train_random_forest.py ^
  --data data_normalized\SI206_2025-05-08\features_worms.csv ^
  --target total_frames ^
  --model-out ml\models\rf_total_frames.joblib

Previous model runs and metrics were cleared due to corrected input data.

Baseline stats (days_alive)
- Output: ml\\analysis\\baseline_days_alive_stats.csv
- SI206_2025-05-08 random_normal: MAE 4.1959, RMSE 5.3847, R2 -3.0099 (train mean 8.6667, std 3.2872)
- SI206_2025-05-08 random_shuffle: MAE 3.1538, RMSE 4.1417, R2 -1.3723 (train pool 48)
- SI216_2025-04-12 random_normal: MAE 3.9977, RMSE 5.1066, R2 -1.9712 (train mean 8.1000, std 2.8966)
- SI216_2025-04-12 random_shuffle: MAE 2.8182, RMSE 4.1231, R2 -0.9369 (train pool 40)
- Combined random_normal: MAE 3.4182, RMSE 4.0905, R2 -1.2206 (train mean 8.2697, std 3.1365)
- Combined random_shuffle: MAE 3.7391, RMSE 4.2014, R2 -1.3427 (train pool 89)

Random Forest (SI206 only, days_alive target)
- Dataset: data_normalized\\SI206_2025-05-08\\features_days_alive.csv
- Features: all engineered features excluding stage_1_frames and stage_5_frames
- Split: 80% train / 20% test, 5-fold CV on train split, n_jobs=1
- Validation (CV mean ± std): MAE 3.0675 ± 0.4198, RMSE 3.8911 ± 0.4405, R2 -0.6422 ± 0.2448
- Test (holdout): MAE 2.0777, RMSE 2.6105, R2 0.0576

Random Forest (Combined SI206 + SI216, days_alive target)
- Dataset: data_normalized\\combined_features_days_alive.csv
- Features: all engineered features excluding stage_1_frames and stage_5_frames
- Split: 80% train / 20% test, 5-fold CV on train split, n_jobs=1
- Validation (CV mean ± std): MAE 2.5869 ± 0.3300, RMSE 3.3476 ± 0.2650, R2 -0.2467 ± 0.1786
- Test (holdout): MAE 2.4361, RMSE 3.0204, R2 -0.2107

Random Forest (Train SI216, Test SI206)
- Train: data_normalized\\SI216_2025-04-12\\features_days_alive.csv
- Test: data_normalized\\SI206_2025-05-08\\features_days_alive.csv
- Features: all engineered features excluding stage_1_frames, stage_6_frames, and total_frames
- Validation (CV mean ± std): MAE 2.4989 ± 0.2288, RMSE 2.9981 ± 0.1365, R2 -0.2057 ± 0.2717
- Test (holdout): MAE 2.7061, RMSE 3.4278, R2 -0.1612

Random Forest (Train SI206, Test SI216)
- Train: data_normalized\\SI206_2025-05-08\\features_days_alive.csv
- Test: data_normalized\\SI216_2025-04-12\\features_days_alive.csv
- Features: all engineered features excluding stage_1_frames, stage_6_frames, and total_frames
- Validation (CV mean ± std): MAE 2.6180 ± 0.3932, RMSE 3.4957 ± 0.5476, R2 -0.4056 ± 0.2913
- Test (holdout): MAE 2.5516, RMSE 2.9708, R2 -0.0357

Reasoning and interpretation of current results
Why the Random Forest is better than "guessing"

A simple "guessing" baseline is to predict the same value for every worm (typically the training mean lifespan). By definition, this baseline captures no feature-driven signal and usually yields R² ≈ 0 (or negative on a test set due to sampling noise).

In our results, the Random Forest reduces absolute error relative to these baselines:

On SI206 (days_alive), the Random Forest achieves Test MAE ≈ 2.08 days with R² ≈ 0.06.

The baseline runs show substantially worse behavior, including negative R² and higher errors in multiple settings.

Even though R² is close to zero, the lower MAE/RMSE indicates the model is extracting some weak predictive signal from the engineered features (i.e., it is not purely guessing). With small datasets, its common for MAE to improve while R² remains near zero because explained variance is hard to estimate reliably from a small test split.

Why adding more worms (combining SI206 + SI216) didn’t improve much

Combining the two experiments increases sample size, but it did not yield a clear generalization gain:

Combined Test R² remains negative (≈ −0.21) and errors are similar to (or worse than) the SI206-only model.

This is consistent with two likely factors:

Batch / experiment effects
The two datasets may differ in acquisition conditions or processing (e.g., imaging setup, illumination, segmentation quality, temperature, food/bacteria conditions, or labeling/stage boundary behavior). These differences can shift feature distributions so that patterns learned in one experiment do not transfer cleanly to the other.

Small-n regime + noisy target
With only ~40–50 worms per experiment, the model is still in a high-variance setting (especially with many engineered features). Lifespan/“days_alive” is also a noisy biological outcome, so more samples help only if they are consistent and comparable across batches.

In other words, more samples only help if they add aligned signal; if they add heterogeneity, the model may not improve (or may degrade) because it is trying to fit conflicting patterns.

Regular vs shuffled-control statistical comparisons

Using the 100 matched regular multirun seeds, one-sided paired Wilcoxon signed-rank tests were computed to test whether the regular-feature models outperform two controls:
- fully shuffled target training (`*_shuffled_metrics.csv`)
- train-on-shuffled / test-on-regular (`*_train_shuffled_test_regular_metrics.csv`)

P-value calculation details:
- Runs were matched by `run` / `seed` between the regular dataset and the control dataset.
- For each metric, a paired delta was computed for every matched run: `delta = regular - control`.
- For `R2`, the one-sided alternative tested whether `delta > 0` (regular higher is better).
- For `MAE` and `RMSE`, the one-sided alternative tested whether `delta < 0` (regular lower is better).
- Reported effect sizes are the mean and median of those paired deltas.

| Comparison | Metric | p-value | Mean delta (regular - control) | Median delta (regular - control) |
|---|---|---:|---:|---:|
| SI206 vs shuffled | R2 | 0.000000000000000002 | 0.8533 | 0.8581 |
| SI206 vs shuffled | MAE | 0.000000000000000002 | -1.1948 | -1.1846 |
| SI206 vs shuffled | RMSE | 0.000000000000000002 | -1.4186 | -1.4235 |
| SI206 vs train_shuffled_test_regular | R2 | 0.000000000000000002 | 0.2317 | 0.2323 |
| SI206 vs train_shuffled_test_regular | MAE | 0.000000000000000002 | -0.8413 | -0.8449 |
| SI206 vs train_shuffled_test_regular | RMSE | 0.000000000000000002 | -0.8412 | -0.8408 |
| SI216 vs shuffled | R2 | 0.000000000000000002 | 1.1359 | 1.1360 |
| SI216 vs shuffled | MAE | 0.000000000000000002 | -1.4653 | -1.4638 |
| SI216 vs shuffled | RMSE | 0.000000000000000002 | -1.3630 | -1.3554 |
| SI216 vs train_shuffled_test_regular | R2 | 0.000000000000000002 | 0.6646 | 0.6631 |
| SI216 vs train_shuffled_test_regular | MAE | 0.000000000000000002 | -0.7892 | -0.7859 |
| SI216 vs train_shuffled_test_regular | RMSE | 0.000000000000000002 | -0.9129 | -0.9101 |
| Combined vs shuffled | R2 | 0.501372 | 0.0002 | -0.0072 |
| Combined vs shuffled | MAE | 0.000000000000000002 | -0.5079 | -0.5082 |
| Combined vs shuffled | RMSE | 0.000000000000000002 | -0.6396 | -0.6345 |
| Combined vs train_shuffled_test_regular | R2 | 0.000000000000000002 | 0.3235 | 0.3256 |
| Combined vs train_shuffled_test_regular | MAE | 0.000000000000000002 | -0.6689 | -0.6692 |
| Combined vs train_shuffled_test_regular | RMSE | 0.000000000000000002 | -0.7827 | -0.7848 |

Interpretation:
- For `SI206` and `SI216`, the regular models are clearly better than both negative controls on all three metrics. The p-values are effectively zero at this scale, and the effect sizes are large.
- For `Combined` versus `train_shuffled_test_regular`, the regular model is also clearly better on `R2`, `MAE`, and `RMSE`, again with large effect sizes and extremely small p-values.
- For `Combined` versus fully `shuffled`, `MAE` and `RMSE` still show a strong and significant advantage for the regular model, but `R2` does not (`p = 0.501372`). That means the combined model is reducing absolute error relative to the shuffled control, but this advantage is not showing up as a stable improvement in explained variance across runs.
- Practically, this supports the claim that the real features contain signal beyond shuffled controls, while also showing that `R2` is the least stable metric here, especially on the combined dataset.

Notes
- `total_frames` is derived from stage boundaries and can serve as a proxy
  target if no explicit lifespan label exists.

Dependencies
- pandas
- numpy
- scikit-learn
- joblib
