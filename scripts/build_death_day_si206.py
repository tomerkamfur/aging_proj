import csv
from pathlib import Path


EXPERIMENT_ID = "SI206_2025-05-08"
BASE_DIR = Path("data_normalized") / EXPERIMENT_ID
FRAME_SECONDS = 3
SECONDS_PER_DAY = 24 * 60 * 60


def load_stage_frames(path: Path):
    frames = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            worm_id = row["worm_id"]
            stage_idx = int(row["stage_index"])
            duration = int(row["end_frame"]) - int(row["start_frame"]) + 1
            frames.setdefault(worm_id, {})[stage_idx] = duration
    return frames


def main():
    stage_info = load_stage_frames(BASE_DIR / "stage_boundaries.csv")
    out_path = BASE_DIR / "death_day.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["worm_id", "total_frames", "total_seconds", "death_day"]
        )
        for worm_id in sorted(stage_info.keys()):
            stages = stage_info[worm_id]
            total_frames = sum(stages.get(i, 0) for i in range(1, 7))
            total_seconds = total_frames * FRAME_SECONDS
            death_day = total_seconds / SECONDS_PER_DAY
            writer.writerow([worm_id, total_frames, total_seconds, death_day])


if __name__ == "__main__":
    main()
