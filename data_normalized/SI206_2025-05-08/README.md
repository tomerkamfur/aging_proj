SI206 normalized outputs

Contents
- metadata_worms.csv: stable worm IDs mapped from sorted *-stages.mat files.
- stage_boundaries.csv: per-worm stage boundaries (stage_index 1..6).
- death_day.csv: per-worm total frames, seconds, and death day (3s per frame).
- time_roaming.csv: roaming_fraction per worm and bin index.
- features_worms.csv: per-worm engineered features (raw scale).
- features_worms_zscore.csv: per-worm engineered features (z-scored).
- features_stage1_3_death_day.csv: early-stage features (1-3) plus death_day.
- features_stage1_3_early_behavior_death_day.csv: stage 1-3 features plus early-only speed/roaming summaries.
- speed_1s/: per-worm npz files, data shape (N, 3).
- speed_10s_avg/: per-worm npz files, data shape (N, 3).
- speed_av2/: per-worm npz files, data shape (N, 6).

Notes
- Stage labels are kept as stage_1..stage_6 to avoid assumptions.
- Worm IDs are derived from camera and well in the stage filename (cam###_w#).
- Speed arrays are stored as npz (compressed) to avoid multi-GB CSVs.
- Column meanings for speed arrays are unknown; columns are labeled col_1..col_N.
