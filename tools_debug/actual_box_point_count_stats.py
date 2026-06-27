#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

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

    local_x = shifted[:, 0] * c + shifted[:, 1] * s
    local_y = -shifted[:, 0] * s + shifted[:, 1] * c
    local_z = shifted[:, 2]

    inside = (
        (np.abs(local_x) <= length / 2.0) &
        (np.abs(local_y) <= width / 2.0) &
        (np.abs(local_z) <= height / 2.0 + z_margin)
    )

    return int(inside.sum())


def bucket_count(n):
    if n == 0:
        return "zero"
    if 1 <= n <= 5:
        return "low_1_to_5"
    return "valid_gt_5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="data/processed/frame_indices/train_frames.csv")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--output-csv", default="outputs/metrics/actual_box_point_counts_train_50.csv")
    args = parser.parse_args()

    dataset = TUMTrafDataset(args.index_path)

    n_frames = min(args.max_frames, len(dataset))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    bucket_counter = Counter()
    bucket_by_class = defaultdict(Counter)

    rows = []

    for idx in tqdm(range(n_frames), desc="Counting points inside boxes", unit="frame"):
        sample = dataset[idx]
        points_xyz = sample["points"][:, :3]

        for box in sample["boxes"]:
            actual_count = count_points_inside_box(points_xyz, box)
            bucket = bucket_count(actual_count)
            cls = box["class_name"]

            class_counter[cls] += 1
            bucket_counter[bucket] += 1
            bucket_by_class[cls][bucket] += 1

            rows.append(
                {
                    "frame_idx": idx,
                    "frame_id": sample["frame_id"],
                    "track_id": box["track_id"],
                    "class_name": cls,
                    "meta_num_points": box.get("num_points"),
                    "actual_points_inside": actual_count,
                    "bucket": bucket,
                }
            )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "frame_id",
                "track_id",
                "class_name",
                "meta_num_points",
                "actual_points_inside",
                "bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_boxes = len(rows)

    print()
    print("=" * 80)
    print(f"Actual point-in-box stats over {n_frames} frames")
    print("=" * 80)
    print(f"Total boxes: {total_boxes}")
    print(f"Saved CSV: {output_csv}")

    print()
    print("Overall buckets:")
    for bucket in ["zero", "low_1_to_5", "valid_gt_5"]:
        count = bucket_counter[bucket]
        pct = 100.0 * count / max(total_boxes, 1)
        print(f"  {bucket:12s}: {count:6d} / {total_boxes:6d} ({pct:5.1f}%)")

    print()
    print("By class:")
    for cls, total_cls in class_counter.most_common():
        z = bucket_by_class[cls]["zero"]
        l = bucket_by_class[cls]["low_1_to_5"]
        v = bucket_by_class[cls]["valid_gt_5"]

        print(
            f"  {cls:18s} total={total_cls:5d} | "
            f"zero={z:4d} ({100*z/max(total_cls,1):5.1f}%) | "
            f"low={l:4d} ({100*l/max(total_cls,1):5.1f}%) | "
            f"valid={v:4d} ({100*v/max(total_cls,1):5.1f}%)"
        )


if __name__ == "__main__":
    main()
