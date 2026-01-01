import csv
from pathlib import Path


EXPERIMENT_ID = "SI206_2025-05-08"
BASE_DIR = Path("data_normalized") / EXPERIMENT_ID


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    features = load_csv(BASE_DIR / "features_worms.csv")
    death_day = load_csv(BASE_DIR / "death_day.csv")
    death_map = {row["worm_id"]: row for row in death_day}

    out_path = BASE_DIR / "features_stage1_3_death_day.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "worm_id",
        "cam_id",
        "well_id",
        "recording_id",
        "timestamp",
        "experiment_id",
        "stage_1_frames",
        "stage_2_frames",
        "stage_3_frames",
        "stage_1_prop",
        "stage_2_prop",
        "stage_3_prop",
        "stage_1_3_frames",
        "death_day",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in features:
            worm_id = row["worm_id"]
            death_row = death_map.get(worm_id)
            if not death_row:
                continue
            stage_1 = int(float(row["stage_1_frames"]))
            stage_2 = int(float(row["stage_2_frames"]))
            stage_3 = int(float(row["stage_3_frames"]))
            stage_1_3 = stage_1 + stage_2 + stage_3

            writer.writerow(
                {
                    "worm_id": worm_id,
                    "cam_id": row["cam_id"],
                    "well_id": row["well_id"],
                    "recording_id": row["recording_id"],
                    "timestamp": row["timestamp"],
                    "experiment_id": row["experiment_id"],
                    "stage_1_frames": stage_1,
                    "stage_2_frames": stage_2,
                    "stage_3_frames": stage_3,
                    "stage_1_prop": float(row["stage_1_prop"]),
                    "stage_2_prop": float(row["stage_2_prop"]),
                    "stage_3_prop": float(row["stage_3_prop"]),
                    "stage_1_3_frames": stage_1_3,
                    "death_day": float(death_row["death_day"]),
                }
            )


if __name__ == "__main__":
    main()
