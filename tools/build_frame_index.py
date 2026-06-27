#!/usr/bin/env python3

from pathlib import Path
import csv
import argparse


def build_index(data_root: Path, split: str, sensor: str):
    labels_dir = data_root / split / "labels_point_clouds" / sensor
    pcd_dir = data_root / split / "point_clouds" / sensor

    label_files = sorted(labels_dir.glob("*.json"))

    rows = []
    missing = []

    for label_path in label_files:
        frame_id = label_path.stem
        pcd_path = pcd_dir / f"{frame_id}.pcd"

        if pcd_path.exists():
            rows.append({
                "split": split,
                "frame_id": frame_id,
                "label_path": str(label_path),
                "point_cloud_path": str(pcd_path),
            })
        else:
            missing.append(str(label_path))

    return rows, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/tumtraf_v2x")
    parser.add_argument(
        "--sensor",
        default="s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path("data/processed/frame_indices")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        rows, missing = build_index(data_root, split, args.sensor)

        output_path = output_dir / f"{split}_frames.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "frame_id", "label_path", "point_cloud_path"]
            )
            writer.writeheader()
            writer.writerows(rows)

        print("=" * 70)
        print(f"Split: {split}")
        print(f"Matched frames: {len(rows)}")
        print(f"Missing point clouds: {len(missing)}")
        print(f"Saved: {output_path}")

        if missing:
            print("First missing examples:")
            for item in missing[:5]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
