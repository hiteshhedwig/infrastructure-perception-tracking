#!/usr/bin/env python3

from pathlib import Path
import json
import math
import numpy as np
import matplotlib.pyplot as plt


LABEL_PATH = Path(
    "data/raw/tumtraf_v2x/train/labels_point_clouds/"
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/"
    "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json"
)

OUTPUT_PATH = Path("outputs/bev/one_frame_boxes_only_bev.png")


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

    rotated = local_corners @ rot.T
    rotated[:, 0] += x
    rotated[:, 1] += y

    return rotated


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    boxes = parse_label(LABEL_PATH)

    print(f"Loaded boxes: {len(boxes)}")

    plt.figure(figsize=(10, 10))

    all_corners = []

    for box in boxes:
        corners = box_corners_bev(box)
        all_corners.append(corners)

        closed = np.vstack([corners, corners[0]])

        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.5)

        label = box["class_name"]
        short_id = box["track_id"][:4]

        plt.text(
            box["x"],
            box["y"],
            f"{label}-{short_id}",
            fontsize=6,
            ha="center",
            va="center",
        )

    all_corners = np.vstack(all_corners)

    margin = 10.0
    plt.xlim(all_corners[:, 0].min() - margin, all_corners[:, 0].max() + margin)
    plt.ylim(all_corners[:, 1].min() - margin, all_corners[:, 1].max() + margin)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("TUMTraf V2X - 3D Boxes Only BEV")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved box BEV image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
