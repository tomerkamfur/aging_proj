ML workspace

This folder contains model training code and artifacts for the aging project.

Getting started
- Input features: `data_normalized/SI206_2025-05-08/features_worms.csv`
- Z-scored features: `data_normalized/SI206_2025-05-08/features_worms_zscore.csv`

Baseline model
- `train_random_forest.py` trains a Random Forest regressor.
- You must specify the target column name from the feature table.

Dataset stats
- SI206 death_day mean: 22.1576 days
- SI206 death_day median: 22.2374 days
- SI206 death_day std: 1.0084 days

Correlation check (SI206 death_day vs early stage duration)
- Output: ml\\analysis\\stage_duration_correlations.csv
- stage_1_frames: Pearson r=0.737 (p=1.09e-13), Spearman r=0.709 (p=2.32e-12)
- stage_2_frames: Pearson r=0.155 (p=0.191), Spearman r=0.207 (p=0.0784)
- stage_3_frames: Pearson r=0.375 (p=0.00106), Spearman r=0.337 (p=0.0036)
- stage_1_2_frames: Pearson r=0.918 (p=2.96e-30), Spearman r=0.908 (p=1.57e-28)
- stage_1_3_frames: Pearson r=0.928 (p=3.04e-32), Spearman r=0.951 (p=7.75e-38)

Baseline results (random normal guess)
- Script: scripts\\baseline_random_normal.py
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Split: 80% train / 20% test, random_state=42
- Train mean/std: 22.0789 / 0.9677
- Test: MAE 1.0632, RMSE 1.2873, R2 -0.4485

Baseline results (random shuffle from train pool)
- Script: scripts\\baseline_random_shuffle.py
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Split: 80% train / 20% test, random_state=42
- Train pool size: 58
- Test: MAE 0.7598, RMSE 1.1275, R2 -0.1112

Example
python ml/train_random_forest.py ^
  --data data_normalized\\SI206_2025-05-08\\features_worms.csv ^
  --target total_frames ^
  --model-out ml\\models\\rf_total_frames.joblib

Baseline results (SI206, early-only features, no proportion features)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Validation (CV mean ± std): MAE 0.2426 ± 0.0879, RMSE 0.3396 ± 0.1393, R2 0.8505 ± 0.0206
- Test (holdout): MAE 0.2808, RMSE 0.4553, R2 0.8188

Baseline results (early roaming + stage 1-3 frames only)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Features: stage_1_frames, stage_2_frames, stage_3_frames, stage_1_3_frames, early_roam_*
- Validation (CV mean ± std): MAE 0.2546 ± 0.0942, RMSE 0.3546 ± 0.1378, R2 0.8220 ± 0.1034
- Test (holdout): MAE 0.2728, RMSE 0.4412, R2 0.8299

Baseline results (stage_1_2_frames, no stage_3_frames)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Features: stage_1_frames, stage_2_frames, stage_1_2_frames, early_roam_*
- Validation (CV mean ± std): MAE 0.3363 ± 0.1038, RMSE 0.4272 ± 0.1378, R2 0.7298 ± 0.1457
- Test (holdout): MAE 0.3035, RMSE 0.4717, R2 0.8055

Baseline results (stage_2_3_frames, no stage_1_2_frames or stage_1_3_frames)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Features: stage_1_frames, stage_2_frames, stage_3_frames, stage_2_3_frames, early_roam_*
- Validation (CV mean ± std): MAE 0.3705 ± 0.1551, RMSE 0.5122 ± 0.2137, R2 0.6423 ± 0.1828
- Test (holdout): MAE 0.4516, RMSE 0.7516, R2 0.5062

Baseline results (stage_2_3_frames, no stage_1_frames)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Features: stage_2_frames, stage_3_frames, stage_2_3_frames, early_roam_*
- Validation (CV mean ± std): MAE 0.7687 ± 0.2431, RMSE 0.9765 ± 0.2858, R2 -0.5797 ± 1.0160
- Test (holdout): MAE 0.8399, RMSE 1.2379, R2 -0.3395

Baseline results (early roaming + stage 1-3 or frames, no stage_1_3_frames or stage_1_2_frames)
- Dataset: data_normalized\\SI206_2025-05-08\\features_stage1_3_early_behavior_death_day.csv
- Target: death_day
- Split: 80% train / 20% test, 5-fold CV on train split
- Features: stage_1_frames, stage_2_frames, stage_3_frames, early_roam_*
- Validation (CV mean ± std): MAE 0.4068 ± 0.1637, RMSE 0.5444 ± 0.2252, R2 0.5713 ± 0.2845
- Test (holdout): MAE 0.4348, RMSE 0.7109, R2 0.5583


Notes
- `total_frames` is derived from stage boundaries and can serve as a proxy
  target if no explicit lifespan label exists.
- Use care to avoid leakage if you later add explicit lifespan labels.

Dependencies
- pandas
- numpy
- scikit-learn
- joblib
