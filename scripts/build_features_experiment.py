import argparse
import csv
from pathlib import Path

import numpy as np


STATS = [
    ("mean", np.mean),
    ("std_over_mean", lambda x: (float(np.std(x) / np.mean(x)) if float(np.mean(x)) != 0.0 else 0.0)),
    # ("std", np.std),
    # ("median", np.median),
    # ("p10", lambda x: np.percentile(x, 10)),
    # ("p90", lambda x: np.percentile(x, 90)),
]

SPEED_STAGE_BUCKETS = (2, 3, 4, 5)
STAGE_ENCODED_SPEED_PREFIXES = {"speed_1s", "speed_10s_avg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-worm features for an experiment.")
    parser.add_argument("--base-dir", required=True, help="Experiment folder in data_normalized.")
    parser.add_argument(
        "--include-av2",
        action="store_true",
        help="Include speed_av2 features if present.",
    )
    return parser.parse_args()


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


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
        feats["roam_slope"] = float(np.polyfit(x, arr, 1)[0])
    else:
        feats["roam_slope"] = 0.0
    return feats


def speed_features(worm_id: str, speed_dir: Path, column_count: int, prefix: str):
    feats = {}
    path = speed_dir / f"{worm_id}.npz"
    data = np.load(path)["data"]
    feats[f"{prefix}_rows"] = int(data.shape[0])

    if prefix in STAGE_ENCODED_SPEED_PREFIXES:
        stage_col = data[:, 2].astype(float)
        for stage_idx in SPEED_STAGE_BUCKETS:
            stage_mask = stage_col == float(stage_idx)
            stage_data = data[stage_mask]
            feats[f"{prefix}_stage{stage_idx}_rows"] = int(stage_data.shape[0])
            for col_idx in (0, 1):
                col = stage_data[:, col_idx].astype(float) if stage_data.size else np.array([])
                for stat_name, value in summarize_array(col).items():
                    feats[f"{prefix}_stage{stage_idx}_col{col_idx + 1}_{stat_name}"] = value
        return feats

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
    args = parse_args()
    base_dir = Path(args.base_dir)
    metadata = load_csv(base_dir / "metadata_worms.csv")
    stage_info = load_stage_features(base_dir / "stage_boundaries.csv")
    roaming = load_roaming(base_dir / "time_roaming.csv")

    speed_sets = [
        ("speed_1s", 3),
        ("speed_10s_avg", 3),
    ]
    # if args.include_av2 and (base_dir / "speed_av2").exists():
    #     speed_sets.append(("speed_av2", 6))

    rows = []
    field_order = []
    for meta in metadata:
        worm_id = meta["worm_id"]
        durations = stage_info.get(worm_id, {})
        total_frames = sum(durations.values())
        row = {
            "worm_id": worm_id,
            "cam_id": meta["cam_id"],
            "well_id": meta["well_id"],
            "recording_id": meta["recording_id"],
            "timestamp": meta["timestamp"],
            "experiment_id": meta["experiment_id"],
            "stage_2_frames": int(durations.get(2, 0)),
            "stage_3_frames": int(durations.get(3, 0)),
            "stage_4_frames": int(durations.get(4, 0)),
            "stage_5_frames": int(durations.get(5, 0)),
        }
        row.update(roaming_features(roaming.get(worm_id, np.array([]))))

        for speed_name, cols in speed_sets:
            row.update(speed_features(worm_id, base_dir / speed_name, cols, speed_name))

        if not field_order:
            field_order = list(row.keys())
        rows.append(row)

    write_csv(base_dir / "features_worms.csv", rows, field_order)


if __name__ == "__main__":
    main()
