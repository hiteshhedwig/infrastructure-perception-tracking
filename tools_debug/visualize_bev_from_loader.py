#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tumtraf_dataset import TUMTrafDataset


def box_corners_bev(box):
    center = box["center"]
    dims = box["dims_lwh"]
    yaw = box["yaw"]

    x, y = center[0], center[1]
    length, width = dims[0], dims[1]

    local = np.array(
        [
            [ length / 2.0,  width / 2.0],
            [ length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0,  width / 2.0],
        ]
    )

    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)],
        ]
    )

    corners = local @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y

    return corners


def draw_bev(sample, output_path):
    points = sample["points"]
    boxes = sample["boxes"]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    mask = (
        (z > -9.0) & (z < 2.0) &
        (x > -80.0) & (x < 90.0) &
        (y > -80.0) & (y < 120.0)
    )

    plt.figure(figsize=(10, 10))

    plt.scatter(
        x[mask],
        y[mask],
        c=intensity[mask],
        s=0.2,
        cmap="gray",
        marker="."
    )

    for box in boxes:
        corners = box_corners_bev(box)
        closed = np.vstack([corners, corners[0]])

        num_points = box.get("num_points")

        if isinstance(num_points, (int, float)) and num_points <= 5:
            plt.plot(closed[:, 0], closed[:, 1], linewidth=1.7, linestyle="--")
        else:
            plt.plot(closed[:, 0], closed[:, 1], linewidth=1.2)

        plt.text(
            box["center"][0],
            box["center"][1],
            box["class_name"],
            fontsize=5,
            ha="center",
            va="center",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{sample['frame_id']} | boxes={len(boxes)}")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.xlim(-80, 90)
    plt.ylim(-80, 120)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-path",
        default="data/processed/frame_indices/train_frames.csv",
    )
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="outputs/bev/from_loader",
    )
    args = parser.parse_args()

    dataset = TUMTrafDataset(args.index_path)

    sample = dataset[args.sample_idx]

    output_path = Path(args.output_dir) / f"{args.sample_idx:04d}_{sample['frame_id']}.png"

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Frame ID: {sample['frame_id']}")
    print(f"Points: {sample['points'].shape}")
    print(f"Boxes: {len(sample['boxes'])}")
    print(f"Saving to: {output_path}")

    draw_bev(sample, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
