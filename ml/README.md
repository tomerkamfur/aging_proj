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

Notes
- `total_frames` is derived from stage boundaries and can serve as a proxy
  target if no explicit lifespan label exists.

Dependencies
- pandas
- numpy
- scikit-learn
- joblib
