#!/usr/bin/env python3

from pathlib import Path
import json
import math


LABEL_PATH = Path(
    "data/raw/tumtraf_v2x/train/labels_point_clouds/"
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/"
    "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json"
)


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
                print(f"Skipping object {object_id}: cuboid length is {len(val)}")
                continue

            x, y, z = val[0], val[1], val[2]
            qx, qy, qz, qw = val[3], val[4], val[5], val[6]
            length, width, height = val[7], val[8], val[9]

            # Since qx and qy are usually zero here, this gives yaw around vertical axis.
            yaw = 2.0 * math.atan2(qz, qw)
            yaw_deg = math.degrees(yaw)

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
                    "yaw_rad": yaw,
                    "yaw_deg": yaw_deg,
                }
            )

    return boxes


def main():
    print(f"Parsing label: {LABEL_PATH}")
    boxes = parse_label(LABEL_PATH)

    print(f"\nTotal boxes parsed: {len(boxes)}")

    print("\nFirst 10 boxes:")
    for i, box in enumerate(boxes[:10]):
        print("-" * 60)
        print(f"Index:      {i}")
        print(f"Class:      {box['class_name']}")
        print(f"Track ID:   {box['track_id']}")
        print(f"Center:     x={box['x']:.2f}, y={box['y']:.2f}, z={box['z']:.2f}")
        print(f"Size:       l={box['length']:.2f}, w={box['width']:.2f}, h={box['height']:.2f}")
        print(f"Yaw:        {box['yaw_rad']:.3f} rad / {box['yaw_deg']:.1f} deg")


if __name__ == "__main__":
    main()
