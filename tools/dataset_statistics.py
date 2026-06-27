#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tumtraf_dataset import TUMTrafDataset


def summarize_split(index_path: Path):
    dataset = TUMTrafDataset(index_path)

    class_counter = Counter()
    boxes_per_frame = []

    unknown_counter = Counter()
    zero_counter = Counter()
    low_counter = Counter()
    valid_counter = Counter()

    known_num_points_by_class = defaultdict(list)
    dims_by_class = defaultdict(list)

    for row in tqdm(dataset.rows, desc=f"Parsing {index_path.stem}", unit="frame"):
        label_path = Path(row["label_path"])
        boxes = TUMTrafDataset.parse_openlabel_boxes(label_path)

        boxes_per_frame.append(len(boxes))

        for box in boxes:
            cls = box["class_name"]
            class_counter[cls] += 1
            dims_by_class[cls].append(box["dims_lwh"])

            num_points = box.get("num_points")

            if num_points is None:
                unknown_counter[cls] += 1
                continue

            try:
                num_points = int(num_points)
            except Exception:
                unknown_counter[cls] += 1
                continue

            if num_points == -1:
                unknown_counter[cls] += 1
            elif num_points == 0:
                zero_counter[cls] += 1
                known_num_points_by_class[cls].append(num_points)
            elif 1 <= num_points <= 5:
                low_counter[cls] += 1
                known_num_points_by_class[cls].append(num_points)
            elif num_points > 5:
                valid_counter[cls] += 1
                known_num_points_by_class[cls].append(num_points)
            else:
                unknown_counter[cls] += 1

    return {
        "num_frames": len(dataset.rows),
        "class_counter": class_counter,
        "boxes_per_frame": boxes_per_frame,
        "unknown_counter": unknown_counter,
        "zero_counter": zero_counter,
        "low_counter": low_counter,
        "valid_counter": valid_counter,
        "known_num_points_by_class": known_num_points_by_class,
        "dims_by_class": dims_by_class,
    }


def pct(part, total):
    return 100.0 * part / max(total, 1)


def print_counter_table(title, counter, class_counter):
    print()
    print(title)
    total = sum(counter.values())
    total_boxes = sum(class_counter.values())
    print(f"  total: {total} / {total_boxes} ({pct(total, total_boxes):.2f}%)")

    for cls, cls_total in class_counter.most_common():
        count = counter[cls]
        print(f"  {cls:18s} {count:6d} / {cls_total:6d} ({pct(count, cls_total):5.1f}%)")


def print_split_summary(split_name: str, stats):
    print()
    print("=" * 80)
    print(f"SPLIT: {split_name}")
    print("=" * 80)

    boxes_per_frame = np.array(stats["boxes_per_frame"])

    print(f"Frames: {stats['num_frames']}")
    print(f"Total boxes: {int(boxes_per_frame.sum())}")

    print()
    print("Boxes per frame:")
    print(f"  min:    {boxes_per_frame.min()}")
    print(f"  max:    {boxes_per_frame.max()}")
    print(f"  mean:   {boxes_per_frame.mean():.2f}")
    print(f"  median: {np.median(boxes_per_frame):.2f}")

    print()
    print("Class counts:")
    for cls, count in stats["class_counter"].most_common():
        print(f"  {cls:18s} {count:6d}")

    print_counter_table(
        "Unknown num_points: num_points == -1 or missing",
        stats["unknown_counter"],
        stats["class_counter"],
    )

    print_counter_table(
        "Zero-point boxes: num_points == 0",
        stats["zero_counter"],
        stats["class_counter"],
    )

    print_counter_table(
        "Low-point boxes: 1 <= num_points <= 5",
        stats["low_counter"],
        stats["class_counter"],
    )

    print_counter_table(
        "Valid-point boxes: num_points > 5",
        stats["valid_counter"],
        stats["class_counter"],
    )

    print()
    print("Known num_points per object by class, excluding -1:")
    for cls in sorted(stats["known_num_points_by_class"].keys()):
        values = np.array(stats["known_num_points_by_class"][cls])

        if len(values) == 0:
            continue

        print(
            f"  {cls:18s} "
            f"min={values.min():6.0f} "
            f"p25={np.percentile(values, 25):6.0f} "
            f"med={np.median(values):6.0f} "
            f"p75={np.percentile(values, 75):6.0f} "
            f"max={values.max():6.0f}"
        )

    print()
    print("Average dimensions by class: length, width, height")
    for cls in sorted(stats["dims_by_class"].keys()):
        dims = np.vstack(stats["dims_by_class"][cls])
        avg = dims.mean(axis=0)
        print(f"  {cls:18s} l={avg[0]:5.2f}, w={avg[1]:5.2f}, h={avg[2]:5.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/processed/frame_indices")
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        index_path = index_dir / f"{split}_frames.csv"

        if not index_path.exists():
            print(f"Skipping missing index: {index_path}")
            continue

        stats = summarize_split(index_path)
        print_split_summary(split, stats)


if __name__ == "__main__":
    main()
