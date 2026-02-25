import csv
from pathlib import Path


BASE_DIR = Path("data_normalized") / "SI206_2025-05-08"


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def build_features_for_experiment(base_dir: Path):
    features = load_csv(base_dir / "features_worms.csv")
    targets = load_csv(base_dir / "targets_days_alive.csv")
    target_map = {row["worm_id"]: row for row in targets}

    out_path = base_dir / "features_days_alive.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not features:
        raise RuntimeError(f"{base_dir}\\features_worms.csv is empty.")

    fieldnames = list(features[0].keys()) + ["days_alive"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in features:
            worm_id = row["worm_id"]
            target = target_map.get(worm_id)
            if not target:
                continue
            out_row = dict(row)
            out_row["days_alive"] = target["days_alive"]
            writer.writerow(out_row)


def build_combined(out_path: Path, base_dirs):
    rows = []
    for base_dir in base_dirs:
        rows.extend(load_csv(base_dir / "features_days_alive.csv"))
    if not rows:
        raise RuntimeError("No rows found for combined dataset.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main():
    build_features_for_experiment(BASE_DIR)
    si216 = Path("data_normalized") / "SI216_2025-04-12"
    build_features_for_experiment(si216)
    build_combined(
        Path("data_normalized") / "combined_features_days_alive.csv",
        [BASE_DIR, si216],
    )


if __name__ == "__main__":
    main()
