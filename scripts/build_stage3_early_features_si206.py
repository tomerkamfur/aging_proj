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


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_stage_endings(path: Path):
    endings = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            worm_id = row["worm_id"]
            stage_idx = int(row["stage_index"])
            end_frame = int(row["end_frame"])
            endings.setdefault(worm_id, {})[stage_idx] = end_frame
    return endings


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
    feats.update({f"early_roam_{k}": v for k, v in stats.items()})
    if arr.size > 1:
        x = np.arange(1, arr.size + 1, dtype=float)
        feats["early_roam_slope"] = float(np.polyfit(x, arr, 1)[0])
    else:
        feats["early_roam_slope"] = 0.0
    return feats


def speed_features(arr: np.ndarray, column_count: int, prefix: str):
    feats = {f"{prefix}_rows": int(arr.shape[0])}
    for col_idx in range(column_count):
        col = arr[:, col_idx].astype(float)
        for stat_name, value in summarize_array(col).items():
            feats[f"{prefix}_col{col_idx + 1}_{stat_name}"] = value
    return feats


def main():
    features = load_csv(BASE_DIR / "features_stage1_3_death_day.csv")
    death_day = load_csv(BASE_DIR / "death_day.csv")
    stage_endings = load_stage_endings(BASE_DIR / "stage_boundaries.csv")
    death_map = {row["worm_id"]: row for row in death_day}

    out_path = BASE_DIR / "features_stage1_3_early_behavior_death_day.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    fieldnames = []

    for row in features:
        worm_id = row["worm_id"]
        endings = stage_endings.get(worm_id, {})
        end_3 = endings.get(3, 0)
        end_6 = endings.get(6, 0)
        ratio = (end_3 / end_6) if end_6 else 0.0

        out_row = dict(row)
        out_row["early_ratio"] = ratio

        roaming_path = BASE_DIR / "time_roaming.csv"
        # Load roaming per worm on demand to keep memory small.
        roaming_vals = []
        with roaming_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for roam_row in reader:
                if roam_row["worm_id"] == worm_id:
                    roaming_vals.append(float(roam_row["roaming_fraction"]))
        roaming_arr = np.asarray(roaming_vals, dtype=float)
        roam_cutoff = int(round(ratio * roaming_arr.size))
        early_roam = roaming_arr[:roam_cutoff] if roam_cutoff > 0 else np.array([])
        out_row.update(roaming_features(early_roam))

        for speed_name, cols in SPEED_SETS:
            speed_path = BASE_DIR / speed_name / f"{worm_id}.npz"
            data = np.load(speed_path)["data"]
            speed_cutoff = int(round(ratio * data.shape[0]))
            early_speed = data[:speed_cutoff] if speed_cutoff > 0 else data[:0]
            out_row.update(speed_features(early_speed, cols, f"early_{speed_name}"))

        death_row = death_map.get(worm_id)
        if death_row:
            out_row["death_day"] = float(death_row["death_day"])

        if not fieldnames:
            fieldnames = list(out_row.keys())
        rows.append(out_row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
