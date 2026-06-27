#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tumtraf_dataset import TUMTrafDataset


def count_points_inside_box(points_xyz, box, z_margin=0.2):
    center = box["center"]
    length, width, height = box["dims_lwh"]
    yaw = box["yaw"]

    shifted = points_xyz - center.reshape(1, 3)

    c = np.cos(yaw)
    s = np.sin(yaw)

    # Transform world/LiDAR points into the local box frame.
    local_x = shifted[:, 0] * c + shifted[:, 1] * s
    local_y = -shifted[:, 0] * s + shifted[:, 1] * c
    local_z = shifted[:, 2]

    inside = (
        (np.abs(local_x) <= length / 2.0) &
        (np.abs(local_y) <= width / 2.0) &
        (np.abs(local_z) <= height / 2.0 + z_margin)
    )

    return int(inside.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="data/processed/frame_indices/train_frames.csv")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    dataset = TUMTrafDataset(args.index_path)
    sample = dataset[args.sample_idx]

    points_xyz = sample["points"][:, :3]
    boxes = sample["boxes"]

    print(f"Frame: {sample['frame_id']}")
    print(f"Points: {points_xyz.shape[0]}")
    print(f"Boxes: {len(boxes)}")
    print()

    rows = []

    for box in boxes:
        actual_count = count_points_inside_box(points_xyz, box)
        meta_count = box.get("num_points")

        rows.append(
            {
                "class_name": box["class_name"],
                "track_id": box["track_id"],
                "meta_num_points": meta_count,
                "actual_points_inside": actual_count,
            }
        )

    rows = sorted(rows, key=lambda r: r["actual_points_inside"])

    print("Boxes sorted by actual LiDAR points inside:")
    print("-" * 90)
    print(f"{'class':15s} {'track':8s} {'meta':>8s} {'actual_inside':>14s}")
    print("-" * 90)

    for r in rows:
        print(
            f"{r['class_name']:15s} "
            f"{r['track_id'][:8]:8s} "
            f"{str(r['meta_num_points']):>8s} "
            f"{r['actual_points_inside']:14d}"
        )

    quality_counter = Counter()

    for r in rows:
        n = r["actual_points_inside"]

        if n == 0:
            quality_counter["zero"] += 1
        elif 1 <= n <= 5:
            quality_counter["low_1_to_5"] += 1
        else:
            quality_counter["valid_gt_5"] += 1

    print()
    print("Actual point-count summary:")
    for k, v in quality_counter.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
