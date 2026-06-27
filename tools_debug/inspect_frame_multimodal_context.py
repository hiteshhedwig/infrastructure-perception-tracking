#!/usr/bin/env python3

from pathlib import Path
import json


DATA_ROOT = Path("data/raw/tumtraf_v2x")
SPLIT = "train"

FRAME_ID = "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"

LABEL_PATH = (
    DATA_ROOT / SPLIT / "labels_point_clouds" /
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered" /
    f"{FRAME_ID}.json"
)

IMAGES_ROOT = DATA_ROOT / SPLIT / "images"


def timestamp_from_stem(stem: str):
    """
    Example:
    1688625742_123810279_s110_camera_basler_east_8mm
    -> 1688625742.123810279
    """
    parts = stem.split("_")

    if len(parts) < 2:
        return None

    try:
        sec = int(parts[0])
        nsec = int(parts[1])
        return sec + nsec * 1e-9
    except ValueError:
        return None


def print_json_context():
    print("=" * 80)
    print("LABEL JSON CONTEXT")
    print("=" * 80)
    print(f"Label path: {LABEL_PATH}")

    with open(LABEL_PATH, "r") as f:
        data = json.load(f)

    openlabel = data["openlabel"]

    print("\nOpenLABEL keys:")
    print(list(openlabel.keys()))

    coord_systems = openlabel.get("coordinate_systems", {})
    print(f"\nCoordinate systems found: {len(coord_systems)}")
    for name, info in coord_systems.items():
        print(f"\nCoordinate system: {name}")
        if isinstance(info, dict):
            for k, v in info.items():
                if k == "pose_wrt_parent":
                    print(f"  {k}: present")
                else:
                    print(f"  {k}: {v}")

    frames = openlabel.get("frames", {})
    print(f"\nFrames found in JSON: {len(frames)}")

    for frame_key, frame in frames.items():
        print(f"\nFrame key: {frame_key}")
        print("Frame keys:")
        print(list(frame.keys()))

        frame_properties = frame.get("frame_properties", {})
        print("\nFrame properties:")
        if not frame_properties:
            print("  None")
        else:
            for k, v in frame_properties.items():
                print(f"  {k}: {v}")

        objects = frame.get("objects", {})
        print(f"\nObjects: {len(objects)}")

        # Print num_points distribution.
        num_points_values = []

        for _, obj in objects.items():
            object_data = obj.get("object_data", {})
            cuboid = object_data.get("cuboid")

            if isinstance(cuboid, list) and len(cuboid) > 0:
                cuboid = cuboid[0]

            if isinstance(cuboid, dict):
                attrs = cuboid.get("attributes", {})
                for attr_type, attr_list in attrs.items():
                    if isinstance(attr_list, list):
                        for attr in attr_list:
                            if attr.get("name") == "num_points":
                                num_points_values.append(attr.get("val"))

        if num_points_values:
            valid = [v for v in num_points_values if isinstance(v, (int, float))]
            if valid:
                print("\nObject num_points stats:")
                print(f"  min: {min(valid)}")
                print(f"  max: {max(valid)}")
                print(f"  avg: {sum(valid) / len(valid):.1f}")
                print(f"  objects with <= 5 points: {sum(v <= 5 for v in valid)}")
        else:
            print("\nNo num_points attributes found.")


def print_nearest_images():
    print("\n" + "=" * 80)
    print("NEAREST CAMERA IMAGES BY TIMESTAMP")
    print("=" * 80)

    lidar_ts = timestamp_from_stem(FRAME_ID)
    print(f"LiDAR/label frame timestamp: {lidar_ts:.9f}")

    camera_dirs = sorted([p for p in IMAGES_ROOT.iterdir() if p.is_dir()])

    for cam_dir in camera_dirs:
        image_files = sorted(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png")))

        candidates = []

        for img_path in image_files:
            ts = timestamp_from_stem(img_path.stem)
            if ts is None:
                continue

            dt = abs(ts - lidar_ts)
            candidates.append((dt, img_path, ts))

        candidates.sort(key=lambda x: x[0])

        print(f"\nCamera: {cam_dir.name}")
        if not candidates:
            print("  No timestamp-parsable images found.")
            continue

        for dt, img_path, ts in candidates[:3]:
            print(f"  dt={dt:.6f}s | {img_path}")


def main():
    print_json_context()
    print_nearest_images()


if __name__ == "__main__":
    main()
