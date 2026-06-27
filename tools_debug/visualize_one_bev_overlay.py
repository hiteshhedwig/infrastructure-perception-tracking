#!/usr/bin/env python3

from pathlib import Path
import json
import math
import numpy as np
import matplotlib.pyplot as plt


FRAME_ID = "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"

PCD_PATH = Path(
    "data/raw/tumtraf_v2x/train/point_clouds/"
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/"
    f"{FRAME_ID}.pcd"
)

LABEL_PATH = Path(
    "data/raw/tumtraf_v2x/train/labels_point_clouds/"
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/"
    f"{FRAME_ID}.json"
)

OUTPUT_PATH = Path("outputs/bev/one_frame_pointcloud_with_boxes_bev.png")


def load_ascii_pcd(pcd_path: Path) -> np.ndarray:
    data_start_line = None

    with open(pcd_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line.startswith("DATA"):
                data_start_line = line_idx + 1
                break

    if data_start_line is None:
        raise ValueError("Could not find DATA line in PCD file.")

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

    for frame_id, frame in frames.items():
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

    local_corners = np.array(
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

    corners = local_corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y

    return corners


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading PCD:   {PCD_PATH}")
    print(f"Loading label: {LABEL_PATH}")

    points = load_ascii_pcd(PCD_PATH)
    boxes = parse_label(LABEL_PATH)

    print(f"Loaded points: {points.shape[0]}")
    print(f"Loaded boxes:  {len(boxes)}")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    # Crop for readable visualization.
    # We keep the region where the annotation boxes are located.
    mask = (
        (z > -9.0) & (z < 2.0) &
        (x > -80.0) & (x < 90.0) &
        (y > -80.0) & (y < 120.0)
    )

    x_crop = x[mask]
    y_crop = y[mask]
    intensity_crop = intensity[mask]

    print(f"Points after crop: {x_crop.shape[0]}")

    plt.figure(figsize=(10, 10))

    plt.scatter(
        x_crop,
        y_crop,
        c=intensity_crop,
        s=0.2,
        cmap="gray",
        marker="."
    )

    for box in boxes:
        corners = box_corners_bev(box)
        closed = np.vstack([corners, corners[0]])

        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.5)

        # Keep text small because this is only a debug image.
        plt.text(
            box["x"],
            box["y"],
            box["class_name"],
            fontsize=5,
            ha="center",
            va="center",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("TUMTraf V2X - Registered Point Cloud + 3D Boxes BEV")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)

    plt.xlim(-80, 90)
    plt.ylim(-80, 120)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved overlay BEV image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
