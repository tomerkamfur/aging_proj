ML workspace

This folder contains model training code and artifacts for the aging project.

Getting started
- Input features: `data_normalized/SI206_2025-05-08/features_worms.csv`
- Z-scored features: `data_normalized/SI206_2025-05-08/features_worms_zscore.csv`

Baseline model
- `train_random_forest.py` trains a Random Forest regressor.
- You must specify the target column name from the feature table.

Example
python ml/train_random_forest.py ^
  --data data_normalized\\SI206_2025-05-08\\features_worms.csv ^
  --target total_frames ^
  --model-out ml\\models\\rf_total_frames.joblib

Baseline results (SI206, early-only features)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Validation (CV mean ± std): MAE 0.2433 ± 0.0971, RMSE 0.3379 ± 0.1503, R2 0.8553 ± 0.0193
- Test (holdout): MAE 0.2610, RMSE 0.4441, R2 0.8276

Notes
- `total_frames` is derived from stage boundaries and can serve as a proxy
  target if no explicit lifespan label exists.
- Use care to avoid leakage if you later add explicit lifespan labels.

Dependencies
- pandas
- numpy
- scikit-learn
- joblib
