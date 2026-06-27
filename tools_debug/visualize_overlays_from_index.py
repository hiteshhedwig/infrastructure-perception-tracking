#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import math
import random

import numpy as np
import matplotlib.pyplot as plt


def load_ascii_pcd(pcd_path: Path) -> np.ndarray:
    data_start_line = None

    with open(pcd_path, "r") as f:
        for line_idx, line in enumerate(f):
            if line.strip().startswith("DATA"):
                data_start_line = line_idx + 1
                break

    if data_start_line is None:
        raise ValueError(f"Could not find DATA line in: {pcd_path}")

    return np.loadtxt(pcd_path, skiprows=data_start_line)


def get_first_cuboid(object_data):
    cuboid = object_data.get("cuboid")

    if cuboid is None:
        return None

    if isinstance(cuboid, list):
        if len(cuboid) == 0:
            return None
        return cuboid[0]

    return cuboid


def parse_label(label_path: Path):
    with open(label_path, "r") as f:
        data = json.load(f)

    boxes = []
    frames = data["openlabel"]["frames"]

    for _, frame in frames.items():
        objects = frame.get("objects", {})

        for object_id, obj in objects.items():
            object_data = obj.get("object_data", {})
            class_name = object_data.get("type", "UNKNOWN")
            cuboid = get_first_cuboid(object_data)

            if cuboid is None:
                continue

            val = cuboid.get("val", [])

            if len(val) != 10:
                continue

            x, y, z = val[0], val[1], val[2]
            qx, qy, qz, qw = val[3], val[4], val[5], val[6]
            length, width, height = val[7], val[8], val[9]

            yaw = 2.0 * math.atan2(qz, qw)

            boxes.append(
                {
                    "track_id": object_id,
                    "class_name": class_name,
                    "x": x,
                    "y": y,
                    "z": z,
                    "length": length,
                    "width": width,
                    "height": height,
                    "yaw": yaw,
                }
            )

    return boxes


def box_corners_bev(box):
    x = box["x"]
    y = box["y"]
    length = box["length"]
    width = box["width"]
    yaw = box["yaw"]

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


def read_index(index_path: Path):
    rows = []

    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def visualize(row, output_path: Path):
    label_path = Path(row["label_path"])
    pcd_path = Path(row["point_cloud_path"])
    frame_id = row["frame_id"]

    points = load_ascii_pcd(pcd_path)
    boxes = parse_label(label_path)

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
        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{frame_id} | boxes={len(boxes)}")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.xlim(-80, 90)
    plt.ylim(-80, 120)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return len(points), len(boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default="data/processed/frame_indices/train_frames.csv",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/bev/index_sample_overlays",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    output_dir = Path(args.output_dir)

    rows = read_index(index_path)

    print(f"Index file: {index_path}")
    print(f"Total frames in index: {len(rows)}")

    random.seed(args.seed)
    sampled_rows = random.sample(rows, k=min(args.num_samples, len(rows)))

    for i, row in enumerate(sampled_rows):
        output_path = output_dir / f"{i:02d}_{row['frame_id']}.png"

        num_points, num_boxes = visualize(row, output_path)

        print(f"[{i:02d}] {row['frame_id']}")
        print(f"     points: {num_points}")
        print(f"     boxes:  {num_boxes}")
        print(f"     saved:  {output_path}")


if __name__ == "__main__":
    main()
