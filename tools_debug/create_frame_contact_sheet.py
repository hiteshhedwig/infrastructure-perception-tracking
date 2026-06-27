#!/usr/bin/env python3

from pathlib import Path
import json
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


DATA_ROOT = Path("data/raw/tumtraf_v2x")
DEFAULT_SPLIT = "train"
DEFAULT_SENSOR = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
DEFAULT_FRAME_ID = "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"


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


def get_num_points_from_cuboid(cuboid):
    attrs = cuboid.get("attributes", {})

    for attr_type, attr_list in attrs.items():
        if not isinstance(attr_list, list):
            continue

        for attr in attr_list:
            if attr.get("name") == "num_points":
                return attr.get("val")

    return None


def parse_label(label_path: Path):
    with open(label_path, "r") as f:
        data = json.load(f)

    frames = data["openlabel"]["frames"]

    # Each JSON has one frame.
    frame = next(iter(frames.values()))
    frame_properties = frame.get("frame_properties", {})

    image_file_names = frame_properties.get("image_file_names", [])

    boxes = []

    for object_id, obj in frame.get("objects", {}).items():
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
        num_points = get_num_points_from_cuboid(cuboid)

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
                "num_points": num_points,
            }
        )

    return boxes, image_file_names


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


def find_image_path(images_root: Path, image_file_name: str):
    matches = list(images_root.rglob(image_file_name))

    if len(matches) == 0:
        return None

    return matches[0]


def draw_bev(ax, pcd_path: Path, boxes):
    points = load_ascii_pcd(pcd_path)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    mask = (
        (z > -9.0) & (z < 2.0) &
        (x > -80.0) & (x < 90.0) &
        (y > -80.0) & (y < 120.0)
    )

    ax.scatter(
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

        num_points = box["num_points"]

        # Boxes with very few points are suspicious/interesting, so mark them dashed.
        if isinstance(num_points, (int, float)) and num_points <= 5:
            ax.plot(closed[:, 0], closed[:, 1], linewidth=1.8, linestyle="--")
            ax.text(
                box["x"],
                box["y"],
                f"{box['class_name']} np={num_points}",
                fontsize=5,
                ha="center",
                va="center",
            )
        else:
            ax.plot(closed[:, 0], closed[:, 1], linewidth=1.2)

    ax.set_title(f"BEV: point cloud + 3D boxes | boxes={len(boxes)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linewidth=0.3)
    ax.set_xlim(-80, 90)
    ax.set_ylim(-80, 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--sensor", default=DEFAULT_SENSOR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    split_root = DATA_ROOT / args.split

    label_path = (
        split_root / "labels_point_clouds" / args.sensor / f"{args.frame_id}.json"
    )

    pcd_path = (
        split_root / "point_clouds" / args.sensor / f"{args.frame_id}.pcd"
    )

    images_root = split_root / "images"

    if args.output is None:
        output_path = Path("outputs/contact_sheets") / f"{args.frame_id}.png"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Label: {label_path}")
    print(f"PCD:   {pcd_path}")

    boxes, image_file_names = parse_label(label_path)

    print(f"Boxes: {len(boxes)}")
    print(f"Images listed in JSON: {len(image_file_names)}")

    low_point_boxes = [
        b for b in boxes
        if isinstance(b["num_points"], (int, float)) and b["num_points"] <= 5
    ]

    print(f"Boxes with <= 5 LiDAR points: {len(low_point_boxes)}")
    for b in low_point_boxes:
        print(
            f"  {b['class_name']} | num_points={b['num_points']} | "
            f"x={b['x']:.2f}, y={b['y']:.2f}"
        )

    fig = plt.figure(figsize=(18, 12))

    # Big BEV plot.
    ax_bev = fig.add_subplot(2, 3, 1)
    draw_bev(ax_bev, pcd_path, boxes)

    # Five camera images.
    for idx, image_file_name in enumerate(image_file_names[:5]):
        ax = fig.add_subplot(2, 3, idx + 2)

        image_path = find_image_path(images_root, image_file_name)

        if image_path is None:
            ax.text(0.5, 0.5, f"Missing image:\n{image_file_name}", ha="center")
            ax.axis("off")
            continue

        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(image_path.parent.name, fontsize=9)
        ax.axis("off")

    fig.suptitle(args.frame_id, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved contact sheet: {output_path}")


if __name__ == "__main__":
    main()
