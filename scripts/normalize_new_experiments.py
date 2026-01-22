import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


@dataclass
class ExperimentConfig:
    name: str
    raw_dir: Path
    excel_csv: Path
    excel_type: str
    out_dir: Path


STAGE_RE = re.compile(
    r"^coordCAM(?P<cam>\d+)-w(?P<well>\d+)CAM(?P=cam)_(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{6})-stages$"
)


def parse_stage_filename(path: Path):
    match = STAGE_RE.match(path.stem)
    if not match:
        return None
    cam_str = match.group("cam")
    well_str = match.group("well")
    timestamp = match.group("timestamp")
    cam_int = int(cam_str)
    well_int = int(well_str)
    worm_id = f"cam{cam_str.zfill(3)}_w{well_str}"
    recording_id = f"CAM{cam_str}_{timestamp}"
    return {
        "cam_str": cam_str,
        "cam_int": cam_int,
        "well_int": well_int,
        "timestamp": timestamp,
        "worm_id": worm_id,
        "recording_id": recording_id,
    }


def read_excel_mapping(path: Path, excel_type: str):
    df = pd.read_csv(path)
    if excel_type == "si206":
        cols = {
            "Worm #": "worm_number",
            "Plate": "cam",
            "Well": "well",
            "Days alive": "days_alive",
            "valid": "valid",
        }
        df = df.rename(columns=cols)
        df = df[list(cols.values())]
        df["cam"] = pd.to_numeric(df["cam"], errors="coerce")
        df["well"] = pd.to_numeric(df["well"], errors="coerce")
        df["valid"] = pd.to_numeric(df["valid"], errors="coerce").fillna(0).astype(int)
        df["worm_number"] = pd.to_numeric(df["worm_number"], errors="coerce")
        df["days_alive"] = pd.to_numeric(df["days_alive"], errors="coerce")
    elif excel_type == "si216":
        cols = {
            "CAM": "cam",
            "Well": "well",
            "Animal number": "worm_number",
            "Days alive": "days_alive",
            "valid": "valid",
        }
        df = df.rename(columns=cols)
        df = df[list(cols.values())]
        df["cam"] = (
            df["cam"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
            .replace("", np.nan)
        )
        df["cam"] = pd.to_numeric(df["cam"], errors="coerce")
        df["well"] = pd.to_numeric(df["well"], errors="coerce")
        df["valid"] = pd.to_numeric(df["valid"], errors="coerce").fillna(0).astype(int)
        df["worm_number"] = pd.to_numeric(df["worm_number"], errors="coerce")
        df["days_alive"] = pd.to_numeric(df["days_alive"], errors="coerce")
    else:
        raise ValueError(f"Unknown excel_type: {excel_type}")

    df = df[df["valid"] == 1].dropna(subset=["cam", "well", "days_alive"])
    df["cam"] = df["cam"].astype(int)
    df["well"] = df["well"].astype(int)
    return df


def load_mat_array(path: Path, key: str):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat[key]


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_experiment(cfg: ExperimentConfig):
    stage_files = sorted(cfg.raw_dir.glob("*stages.mat"))
    if not stage_files:
        raise RuntimeError(f"No stage files found in {cfg.raw_dir}")

    excel_df = read_excel_mapping(cfg.excel_csv, cfg.excel_type)
    excel_map = {
        (row.cam, row.well): row for row in excel_df.itertuples(index=False)
    }

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    stage_rows = []
    target_rows = []
    valid_indices = []
    valid_worm_ids = []

    for idx, stage_file in enumerate(stage_files):
        parsed = parse_stage_filename(stage_file)
        if not parsed:
            continue
        key = (parsed["cam_int"], parsed["well_int"])
        excel_row = excel_map.get(key)
        if not excel_row:
            continue

        valid_indices.append(idx)
        valid_worm_ids.append(parsed["worm_id"])

        metadata_rows.append(
            {
                "worm_id": parsed["worm_id"],
                "cam_id": parsed["cam_int"],
                "well_id": parsed["well_int"],
                "recording_id": parsed["recording_id"],
                "timestamp": parsed["timestamp"],
                "experiment_id": cfg.name,
                "stage_file": stage_file.name,
                "excel_worm_number": excel_row.worm_number,
                "excel_days_alive": excel_row.days_alive,
            }
        )

        mat = sio.loadmat(stage_file, squeeze_me=True, struct_as_record=False)
        boundaries = np.asarray(mat["boundaries"]).astype(int).tolist()
        file_start = int(np.asarray(mat["file_start"]).item())
        for stage_idx, end_frame in enumerate(boundaries, start=1):
            start_frame = file_start if stage_idx == 1 else boundaries[stage_idx - 2] + 1
            stage_rows.append(
                {
                    "worm_id": parsed["worm_id"],
                    "stage_index": stage_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "file_start": file_start,
                    "source_file": stage_file.name,
                }
            )

        target_rows.append(
            {
                "worm_id": parsed["worm_id"],
                "days_alive": excel_row.days_alive,
            }
        )

    if not valid_indices:
        raise RuntimeError(f"No valid worms found for {cfg.name}")

    write_csv(cfg.out_dir / "metadata_worms.csv", metadata_rows, list(metadata_rows[0].keys()))
    write_csv(cfg.out_dir / "stage_boundaries.csv", stage_rows, list(stage_rows[0].keys()))
    write_csv(cfg.out_dir / "targets_days_alive.csv", target_rows, list(target_rows[0].keys()))

    roaming = load_mat_array(cfg.raw_dir / "individual_time_roaming.mat", "individual_time_roaming")
    roaming = np.asarray(roaming)
    roaming = roaming[valid_indices, :]
    roaming_rows = []
    for worm_id, row in zip(valid_worm_ids, roaming):
        for idx, value in enumerate(row, start=1):
            roaming_rows.append(
                {"worm_id": worm_id, "bin_idx": idx, "roaming_fraction": float(value)}
            )
    write_csv(
        cfg.out_dir / "time_roaming.csv",
        roaming_rows,
        ["worm_id", "bin_idx", "roaming_fraction"],
    )

    speed_sets = [
        ("individual_speed_AV_1s.mat", "individual_speed_AV_1s", "speed_1s", 3),
        ("individual_speed_AV_1s_average_10swindow.mat", "individual_speed_AV_1s_average_10swindow", "speed_10s_avg", 3),
        ("individual_Speed_AV2.mat", "individual_Speed_AV2", "speed_av2", 6),
    ]

    for filename, key, out_name, col_count in speed_sets:
        mat_path = cfg.raw_dir / filename
        if not mat_path.exists():
            continue
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if key not in mat:
            continue
        arr = mat[key]
        if arr.shape[0] != len(stage_files):
            arr = np.asarray(arr)
        out_dir = cfg.out_dir / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for worm_id, idx in zip(valid_worm_ids, valid_indices):
            data = np.asarray(arr[idx])
            columns = np.array([f"col_{i}" for i in range(1, col_count + 1)])
            np.savez_compressed(out_dir / f"{worm_id}.npz", data=data, columns=columns)


def main():
    experiments = [
        ExperimentConfig(
            name="SI206_2025-05-08",
            raw_dir=Path(r"D:\Data for aging project\SI206, N2 Llifespan, 08-05-2025"),
            excel_csv=Path("data_normalized/si206_excel.csv"),
            excel_type="si206",
            out_dir=Path("data_normalized/SI206_2025-05-08"),
        ),
        ExperimentConfig(
            name="SI216_2025-04-12",
            raw_dir=Path(r"D:\Data for aging project\SI216, N2, lifespan, 04-12-2025"),
            excel_csv=Path("data_normalized/si216_excel.csv"),
            excel_type="si216",
            out_dir=Path("data_normalized/SI216_2025-04-12"),
        ),
    ]

    for cfg in experiments:
        normalize_experiment(cfg)
        print(f"Normalized {cfg.name} -> {cfg.out_dir}")


if __name__ == "__main__":
    main()
