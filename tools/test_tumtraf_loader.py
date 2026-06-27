#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
from src.data.tumtraf_dataset import TUMTrafDataset

def main():
    dataset = TUMTrafDataset("data/processed/frame_indices/train_frames.csv")

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print()
    print("First sample:")
    print(f"  frame_id: {sample['frame_id']}")
    print(f"  points shape: {sample['points'].shape}")
    print(f"  boxes: {len(sample['boxes'])}")
    print(f"  label_path: {sample['label_path']}")
    print(f"  point_cloud_path: {sample['point_cloud_path']}")

    class_counts = Counter(box["class_name"] for box in sample["boxes"])

    print()
    print("Classes in first frame:")
    for cls, count in class_counts.most_common():
        print(f"  {cls}: {count}")

    print()
    print("First 5 boxes:")
    for box in sample["boxes"][:5]:
        center = box["center"]
        dims = box["dims_lwh"]

        print(
            f"  {box['class_name']:12s} "
            f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) "
            f"dims_lwh=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}) "
            f"yaw={box['yaw']:.2f} "
            f"num_points={box['num_points']}"
        )


if __name__ == "__main__":
    main()
