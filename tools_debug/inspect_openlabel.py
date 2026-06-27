#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter


def get_attributes(object_data):
    attrs = {}

    for data_key in ["cuboid", "bbox", "poly2d", "point2d"]:
        data_obj = object_data.get(data_key)
        if not data_obj:
            continue

        if isinstance(data_obj, list):
            data_obj = data_obj[0] if data_obj else {}

        attr_block = data_obj.get("attributes", {})
        for attr_type, attr_list in attr_block.items():
            if isinstance(attr_list, list):
                for a in attr_list:
                    name = a.get("name")
                    val = a.get("val")
                    if name is not None:
                        attrs[name] = val

    return attrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=None,
        help="Path to one OpenLABEL JSON file. If omitted, first train label is used.",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/raw/tumtraf_v2x/train/labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered",
    )
    parser.add_argument("--max-objects", type=int, default=5)
    args = parser.parse_args()

    if args.label is None:
        label_path = sorted(Path(args.labels_dir).glob("*.json"))[0]
    else:
        label_path = Path(args.label)

    print(f"Label file: {label_path}")

    with open(label_path, "r") as f:
        data = json.load(f)

    if "openlabel" not in data:
        print("ERROR: This does not look like OpenLABEL format.")
        print("Top-level keys:", list(data.keys()))
        return

    openlabel = data["openlabel"]

    print("\nTop-level OpenLABEL keys:")
    print(list(openlabel.keys()))

    metadata = openlabel.get("metadata", {})
    print("\nMetadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    frames = openlabel.get("frames", {})
    print(f"\nNumber of frames in this JSON: {len(frames)}")

    class_counter = Counter()
    total_objects = 0

    print("\nObjects preview:")

    for frame_id, frame in frames.items():
        print(f"\nFrame ID: {frame_id}")

        objects = frame.get("objects", {})
        print(f"Objects in frame: {len(objects)}")

        for i, (object_id, obj) in enumerate(objects.items()):
            object_data = obj.get("object_data", {})
            obj_type = object_data.get("type", "UNKNOWN")
            class_counter[obj_type] += 1
            total_objects += 1

            cuboid = object_data.get("cuboid", None)

            print("\n------------------------------")
            print(f"Object index: {i}")
            print(f"Object ID / track ID candidate: {object_id}")
            print(f"Class/type: {obj_type}")

            if cuboid is None:
                print("Cuboid: missing")
            else:
                if isinstance(cuboid, list):
                    cuboid = cuboid[0] if cuboid else {}

                val = cuboid.get("val", [])
                print(f"Cuboid val length: {len(val)}")
                print(f"Cuboid val: {val}")

                attrs = get_attributes(object_data)
                print("Attributes:")
                for k, v in attrs.items():
                    print(f"  {k}: {v}")

            if i + 1 >= args.max_objects:
                break

    print("\n================ SUMMARY ================")
    print(f"Total objects seen: {total_objects}")
    print("Class counts:")
    for cls, count in class_counter.most_common():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
