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


