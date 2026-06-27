#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter


def short_value(v, max_len=120):
    s = repr(v)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def print_schema(obj, indent=0, max_depth=4):
    prefix = "  " * indent

    if indent > max_depth:
        print(prefix + "...")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: {type(v).__name__}")
            print_schema(v, indent + 1, max_depth)

    elif isinstance(obj, list):
        print(f"{prefix}[list] length={len(obj)}")
        if len(obj) > 0:
            print(f"{prefix}first item type: {type(obj[0]).__name__}")
            print_schema(obj[0], indent + 1, max_depth)

    else:
        print(f"{prefix}{short_value(obj)}")


def find_possible_objects(obj):
    """
    Heuristic search for object/annotation lists inside unknown JSON structure.
    """
    candidates = []

    def walk(x, path="root"):
        if isinstance(x, dict):
            for k, v in x.items():
                new_path = f"{path}.{k}"
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    candidates.append((new_path, v))
                walk(v, new_path)
        elif isinstance(x, list):
            for i, item in enumerate(x[:3]):
                walk(item, f"{path}[{i}]")

    walk(obj)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-dir",
        default="data/raw/tumtraf_v2x/train/labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered",
    )
    parser.add_argument("--max-files", type=int, default=50)
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    json_files = sorted(labels_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {labels_dir}")

    print(f"Labels dir: {labels_dir}")
    print(f"JSON files found: {len(json_files)}")
    print(f"First JSON: {json_files[0]}")
    print("\n================ FIRST FILE SCHEMA ================\n")

    with open(json_files[0], "r") as f:
        data = json.load(f)

    print_schema(data)

    print("\n================ POSSIBLE OBJECT LISTS ================\n")
    candidates = find_possible_objects(data)

    for path, items in candidates:
        print(f"Candidate path: {path}")
        print(f"Length: {len(items)}")
        print(f"First item keys: {list(items[0].keys())}")
        print()

    print("\n================ CLASS / CATEGORY GUESS ================\n")

    class_counter = Counter()

    for jf in json_files[: args.max_files]:
        with open(jf, "r") as f:
            d = json.load(f)

        candidates = find_possible_objects(d)

        for _, items in candidates:
            for item in items:
                for key in ["class", "category", "type", "name", "object_class"]:
                    if key in item and isinstance(item[key], str):
                        class_counter[item[key]] += 1

    if class_counter:
        for cls, count in class_counter.most_common():
            print(f"{cls}: {count}")
    else:
        print("Could not auto-detect class/category field. Need to inspect schema manually.")


if __name__ == "__main__":
    main()
