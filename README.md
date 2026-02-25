# aging_proj

This repository contains an end-to-end machine learning pipeline to predict **time of death / remaining lifespan** of individual *C. elegans* worms from **longitudinal behavioral tracking** data collected at high temporal resolution.

The project is inspired by the idea that behavior provides a precise “biological clock” signal: in *C. elegans*, developmental stage transitions are strongly marked by **lethargus** (sleep-like inactivity) periods, enabling accurate time-alignment of individuals even when their developmental pace differs.  This repo adapts that philosophy to lifespan prediction: we engineer per-worm behavioral features (development timing + activity state metrics) and train supervised models (starting with **Random Forest** and other baselines) to predict survival outcomes.

### What we predict

Depending on the dataset availability, the pipeline supports:

* **Regression:** predict *time-to-death* (or remaining lifespan) in minutes/hours/days.
* **Survival modeling (optional):** predict risk over time (e.g., Random Survival Forest / Cox baselines) when censoring exists.

### Core features (per individual)

Features are derived from frame-level tracking (e.g., one frame every 3 seconds) and summarized into interpretable per-worm descriptors, including:

**1) Developmental timing**

* Duration (in frames or minutes) of each life stage: **L1, L2, L3, L4, adulthood**
* Timing of stage transitions (absolute or relative to hatching)
* Stage-specific pace metrics (e.g., proportion of life spent per stage)

**2) Locomotion / activity**

* Roaming vs. dwelling statistics across time bins (global and stage-specific)
* Fraction of time roaming/dwelling
* Transition rates (roaming↔dwelling switching frequency)
* Speed/trajectory summaries (mean/median, variability, high-activity bursts)

**3) Time-structured summaries**

* Features computed per stage and/or fixed windows (early life vs late life)
* Trend features (does activity decay with age? how fast?)

### Modeling approach

We start with **tree-based models** because they are robust, handle nonlinear interactions well, and provide feature importance:

* Random Forest Regressor (baseline)
* Gradient boosting (e.g., XGBoost/LightGBM-style equivalents if used)
* Optional: Random Survival Forest for censored survival data

The repo includes careful evaluation to avoid leakage:

* Train/test split by individual (never mixing time bins from the same worm across splits)
* Cross-validation with consistent preprocessing
* Metrics such as MAE/RMSE (regression) and concordance index (survival)

### Feature reference (current engineered features)

Notes:
- `stage_1_frames`, `stage_6_frames`, and `total_frames` are excluded from current feature tables due to data issues, but they may still appear in older outputs.
- Feature names with `col_N` refer to column N in the original MATLAB arrays (the raw files do not provide column labels).
- If you see `col_1` vs `col_10`, they simply mean different columns from the source array (index 1 vs index 10).

**Identifiers**
- `worm_id`: stable ID derived from camera and well (e.g., cam034_w1).
- `cam_id`: camera/plate number from filename.
- `well_id`: well number from filename.
- `recording_id`: camera + timestamp identifier.
- `timestamp`: recording timestamp from filename.
- `experiment_id`: experiment label (e.g., SI206_2025-05-08).

**Stage duration features**
- `stage_2_frames`, `stage_3_frames`, `stage_4_frames`, `stage_5_frames`: frame counts per stage.
- `total_frames`: sum of frame counts across stages.
- (Legacy) `stage_1_frames`, `stage_6_frames`: excluded from current modeling.
- (Legacy) `stage_{i}_prop`: stage proportion = stage_i_frames / total_frames.

**Roaming features (time_roaming)**
- `roam_mean`, `roam_std`, `roam_median`, `roam_p10`, `roam_p90`: summary stats of roaming fraction.
- `roam_early_mean`, `roam_mid_mean`, `roam_late_mean`: mean roaming in first, middle, last third.
- `roam_slope`: linear trend slope of roaming fraction over time.

**Speed features**
Each speed feature set is computed per worm from the raw arrays:
- `speed_1s_*`: stats from `individual_speed_AV_1s.mat` (columns 1..3).
- `speed_10s_avg_*`: stats from `individual_speed_AV_1s_average_10swindow.mat` (columns 1..3).
- `speed_av2_*`: stats from `individual_Speed_AV2.mat` (columns 1..6; SI206 only).

For each set:
- `{prefix}_rows`: number of rows (time points) in the array.
- `{prefix}_col{N}_mean`, `{prefix}_col{N}_std`, `{prefix}_col{N}_median`, `{prefix}_col{N}_p10`, `{prefix}_col{N}_p90`: summary stats for column N.


