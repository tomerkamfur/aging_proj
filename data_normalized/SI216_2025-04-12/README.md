SI216 normalized outputs (from D:\Data for aging project)

Contents
- metadata_worms.csv: worm IDs matched to Excel metadata (valid=1 only).
- stage_boundaries.csv: per-worm stage boundaries (stage_index 1..6).
- targets_days_alive.csv: per-worm days_alive from the Excel sheet.
- time_roaming.csv: roaming_fraction per worm and bin index.
- speed_1s/: per-worm npz files, data shape (N, 3).
- speed_10s_avg/: per-worm npz files, data shape (N, 3).

Notes
- Worm IDs are derived from camera and well in the stage filename (cam###_w#).
- Only rows with valid=1 are included.
- Column meanings for speed arrays are unknown; columns are labeled col_1..col_N.
