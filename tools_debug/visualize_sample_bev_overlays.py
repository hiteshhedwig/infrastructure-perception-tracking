#!/usr/bin/env python3

from pathlib import Path
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt


DATA_ROOT = Path("data/raw/tumtraf_v2x")
SPLIT = "train"
SENSOR = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
OUTPUT_DIR = Path("outputs/bev/sample_overlays")


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

    frames = data["openlabel"]["frames"]
    boxes = []

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


def visualize_frame(label_path: Path, pcd_path: Path, output_path: Path):
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
        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(label_path.stem)
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.xlim(-80, 90)
    plt.ylim(-80, 120)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return len(points), len(boxes)


def main():
    labels_dir = DATA_ROOT / SPLIT / "labels_point_clouds" / SENSOR
    pcd_dir = DATA_ROOT / SPLIT / "point_clouds" / SENSOR

    label_files = sorted(labels_dir.glob("*.json"))

    random.seed(42)
    sample_files = random.sample(label_files, k=min(10, len(label_files)))

    print(f"Generating overlays for {len(sample_files)} frames...")

    for label_path in sample_files:
        frame_id = label_path.stem
        pcd_path = pcd_dir / f"{frame_id}.pcd"

        if not pcd_path.exists():
            print(f"Missing PCD for: {frame_id}")
            continue

        output_path = OUTPUT_DIR / f"{frame_id}.png"
        num_points, num_boxes = visualize_frame(label_path, pcd_path, output_path)

        print(f"{frame_id}")
        print(f"  points: {num_points}")
        print(f"  boxes:  {num_boxes}")
        print(f"  saved:  {output_path}")


if __name__ == "__main__":
    main()
