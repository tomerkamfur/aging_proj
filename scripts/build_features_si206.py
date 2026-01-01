import csv
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "SI206_2025-05-08"
BASE_DIR = Path("data_normalized") / EXPERIMENT_ID

SPEED_SETS = [
    ("speed_1s", 3),
    ("speed_10s_avg", 3),
    ("speed_av2", 6),
]

STATS = [
    ("mean", np.mean),
    ("std", np.std),
    ("median", np.median),
    ("p10", lambda x: np.percentile(x, 10)),
    ("p90", lambda x: np.percentile(x, 90)),
]


def load_metadata(path: Path):
    metadata = {}
    order = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            worm_id = row["worm_id"]
            metadata[worm_id] = row
            order.append(worm_id)
    return metadata, order


def load_stage_features(path: Path):
    stage_info = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            worm_id = row["worm_id"]
            stage_idx = int(row["stage_index"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            duration = end_frame - start_frame + 1
            stage_info.setdefault(worm_id, {})[stage_idx] = duration
    return stage_info


def load_roaming(path: Path):
    roaming = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            worm_id = row["worm_id"]
            roaming.setdefault(worm_id, []).append(float(row["roaming_fraction"]))
    return {k: np.asarray(v, dtype=float) for k, v in roaming.items()}


def summarize_array(arr: np.ndarray):
    out = {}
    if arr.size == 0:
        for name, _ in STATS:
            out[name] = 0.0
        return out
    for name, func in STATS:
        out[name] = float(func(arr))
    return out


def roaming_features(arr: np.ndarray):
    feats = {}
    stats = summarize_array(arr)
    feats.update({f"roam_{k}": v for k, v in stats.items()})

    if arr.size == 0:
        feats["roam_early_mean"] = 0.0
        feats["roam_mid_mean"] = 0.0
        feats["roam_late_mean"] = 0.0
        feats["roam_slope"] = 0.0
        return feats

    third = max(int(arr.size / 3), 1)
    early = arr[:third]
    mid = arr[third : 2 * third]
    late = arr[2 * third :]
    feats["roam_early_mean"] = float(np.mean(early)) if early.size else 0.0
    feats["roam_mid_mean"] = float(np.mean(mid)) if mid.size else 0.0
    feats["roam_late_mean"] = float(np.mean(late)) if late.size else 0.0

    if arr.size > 1:
        x = np.arange(1, arr.size + 1, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
    else:
        slope = 0.0
    feats["roam_slope"] = slope
    return feats


def stage_features(stage_info, worm_id):
    feats = {}
    durations = stage_info.get(worm_id, {})
    total_frames = sum(durations.values())
    for idx in range(1, 7):
        duration = int(durations.get(idx, 0))
        feats[f"stage_{idx}_frames"] = duration
    for idx in range(1, 7):
        duration = int(durations.get(idx, 0))
        feats[f"stage_{idx}_prop"] = float(duration / total_frames) if total_frames else 0.0
    feats["total_frames"] = int(total_frames)
    return feats


def speed_features(worm_id, speed_dir: Path, column_count: int, prefix: str):
    feats = {}
    path = speed_dir / f"{worm_id}.npz"
    data = np.load(path)["data"]
    feats[f"{prefix}_rows"] = int(data.shape[0])
    for col_idx in range(column_count):
        col = data[:, col_idx].astype(float)
        for stat_name, value in summarize_array(col).items():
            feats[f"{prefix}_col{col_idx + 1}_{stat_name}"] = value
    return feats


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    metadata, worm_order = load_metadata(BASE_DIR / "metadata_worms.csv")
    stage_info = load_stage_features(BASE_DIR / "stage_boundaries.csv")
    roaming = load_roaming(BASE_DIR / "time_roaming.csv")

    feature_rows = []
    feature_order = []

    for worm_id in worm_order:
        row = {
            "worm_id": worm_id,
            "cam_id": metadata[worm_id]["cam_id"],
            "well_id": metadata[worm_id]["well_id"],
            "recording_id": metadata[worm_id]["recording_id"],
            "timestamp": metadata[worm_id]["timestamp"],
            "experiment_id": metadata[worm_id]["experiment_id"],
        }

        row.update(stage_features(stage_info, worm_id))
        row.update(roaming_features(roaming.get(worm_id, np.array([]))))

        for speed_name, cols in SPEED_SETS:
            row.update(speed_features(worm_id, BASE_DIR / speed_name, cols, speed_name))

        if not feature_order:
            feature_order = list(row.keys())
        feature_rows.append(row)

    write_csv(BASE_DIR / "features_worms.csv", feature_rows, feature_order)

    numeric_fields = [
        name
        for name in feature_order
        if name
        not in {
            "worm_id",
            "cam_id",
            "well_id",
            "recording_id",
            "timestamp",
            "experiment_id",
        }
    ]

    matrix = np.array([[row[name] for name in numeric_fields] for row in feature_rows], dtype=float)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds[stds == 0] = 1.0
    zscores = (matrix - means) / stds

    z_rows = []
    for worm_id, meta_row, zrow in zip(worm_order, feature_rows, zscores):
        z_entry = {
            "worm_id": worm_id,
            "cam_id": meta_row["cam_id"],
            "well_id": meta_row["well_id"],
            "recording_id": meta_row["recording_id"],
            "timestamp": meta_row["timestamp"],
            "experiment_id": meta_row["experiment_id"],
        }
        for name, value in zip(numeric_fields, zrow):
            z_entry[name] = float(value)
        z_rows.append(z_entry)

    write_csv(BASE_DIR / "features_worms_zscore.csv", z_rows, feature_order)


if __name__ == "__main__":
    main()
