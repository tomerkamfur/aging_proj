import csv
import re
from pathlib import Path

import numpy as np
import scipy.io as sio


EXPERIMENT_ID = "SI206_2025-05-08"
SOURCE_DIR = Path("Data for aging project") / "SI206, N2 Llifespan, 08-05-2025"
OUTPUT_DIR = Path("data_normalized") / EXPERIMENT_ID

STAGE_RE = re.compile(
    r"^coordCAM(?P<cam>\d+)-w(?P<well>\d+)CAM(?P=cam)_(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{6})-stages$"
)


def parse_stage_filename(path: Path) -> dict:
    stem = path.stem
    match = STAGE_RE.match(stem)
    if not match:
        return {
            "cam_id": "",
            "well_id": "",
            "timestamp": "",
            "recording_id": "",
            "worm_id": stem,
        }
    cam_id = match.group("cam")
    well_id = match.group("well")
    timestamp = match.group("timestamp")
    recording_id = f"CAM{cam_id}_{timestamp}"
    worm_id = f"cam{cam_id}_w{well_id}"
    return {
        "cam_id": cam_id,
        "well_id": well_id,
        "timestamp": timestamp,
        "recording_id": recording_id,
        "worm_id": worm_id,
    }


def load_mat_array(path: Path, key: str) -> np.ndarray:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat[key]


def write_stage_boundaries(stage_files, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "worm_id",
                "stage_index",
                "start_frame",
                "end_frame",
                "file_start",
                "source_file",
            ]
        )
        for stage_file in stage_files:
            mat = sio.loadmat(stage_file, squeeze_me=True, struct_as_record=False)
            boundaries = np.asarray(mat["boundaries"]).astype(int).tolist()
            file_start = int(np.asarray(mat["file_start"]).item())
            meta = parse_stage_filename(stage_file)
            for idx, end_frame in enumerate(boundaries, start=1):
                if idx == 1:
                    start_frame = file_start
                else:
                    start_frame = boundaries[idx - 2] + 1
                writer.writerow(
                    [
                        meta["worm_id"],
                        idx,
                        start_frame,
                        end_frame,
                        file_start,
                        stage_file.name,
                    ]
                )


def write_time_roaming(worm_ids, roaming_matrix: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["worm_id", "bin_idx", "roaming_fraction"])
        for worm_id, row in zip(worm_ids, roaming_matrix):
            for idx, value in enumerate(row, start=1):
                writer.writerow([worm_id, idx, float(value)])


def write_metadata(stage_files, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "worm_id",
                "cam_id",
                "well_id",
                "recording_id",
                "timestamp",
                "experiment_id",
                "stage_file",
                "speed_row_index",
            ]
        )
        for idx, stage_file in enumerate(stage_files, start=1):
            meta = parse_stage_filename(stage_file)
            writer.writerow(
                [
                    meta["worm_id"],
                    meta["cam_id"],
                    meta["well_id"],
                    meta["recording_id"],
                    meta["timestamp"],
                    EXPERIMENT_ID,
                    stage_file.name,
                    idx,
                ]
            )


def write_speed_npz(worm_ids, speed_cells, output_dir: Path, column_count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = np.array([f"col_{i}" for i in range(1, column_count + 1)])
    for worm_id, arr in zip(worm_ids, speed_cells):
        out_path = output_dir / f"{worm_id}.npz"
        if out_path.exists():
            continue
        np.savez_compressed(out_path, data=np.asarray(arr), columns=columns)


def main():
    stage_files = sorted(SOURCE_DIR.glob("*stages.mat"))
    if not stage_files:
        raise RuntimeError(f"No stage files found in {SOURCE_DIR}")

    worm_ids = [parse_stage_filename(p)["worm_id"] for p in stage_files]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_metadata(stage_files, OUTPUT_DIR / "metadata_worms.csv")
    write_stage_boundaries(stage_files, OUTPUT_DIR / "stage_boundaries.csv")

    roaming = load_mat_array(SOURCE_DIR / "individual_time_roaming.mat", "individual_time_roaming")
    write_time_roaming(worm_ids, np.asarray(roaming), OUTPUT_DIR / "time_roaming.csv")

    speed_1s = load_mat_array(SOURCE_DIR / "individual_speed_AV_1s.mat", "individual_speed_AV_1s")
    write_speed_npz(worm_ids, speed_1s, OUTPUT_DIR / "speed_1s", 3)

    speed_10s = load_mat_array(
        SOURCE_DIR / "individual_speed_AV_1s_average_10swindow.mat",
        "individual_speed_AV_1s_average_10swindow",
    )
    write_speed_npz(worm_ids, speed_10s, OUTPUT_DIR / "speed_10s_avg", 3)

    speed_av2 = load_mat_array(SOURCE_DIR / "individual_Speed_AV2.mat", "individual_Speed_AV2")
    write_speed_npz(worm_ids, speed_av2, OUTPUT_DIR / "speed_av2", 6)


if __name__ == "__main__":
    main()
