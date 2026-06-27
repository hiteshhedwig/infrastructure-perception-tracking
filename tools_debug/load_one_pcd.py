#!/usr/bin/env python3

from pathlib import Path
import numpy as np


PCD_PATH = Path(
    "data/raw/tumtraf_v2x/train/point_clouds/"
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/"
    "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.pcd"
)


def load_ascii_pcd(pcd_path: Path):
    header = []
    data_start_line = None

    with open(pcd_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            header.append(line)

            if line.startswith("DATA"):
                data_start_line = line_idx + 1
                break

    if data_start_line is None:
        raise ValueError("Could not find DATA line in PCD file.")

    points = np.loadtxt(pcd_path, skiprows=data_start_line)

    return header, points


def main():
    print(f"Loading PCD: {PCD_PATH}")

    header, points = load_ascii_pcd(PCD_PATH)

    print("\nHeader summary:")
    for line in header:
        if line.startswith(("FIELDS", "WIDTH", "HEIGHT", "POINTS", "DATA")):
            print(f"  {line}")

    print("\nLoaded point cloud:")
    print(f"  shape: {points.shape}")

    xyz = points[:, :3]
    intensity = points[:, 3]

    print("\nXYZ ranges:")
    print(f"  x: {xyz[:, 0].min():.3f} to {xyz[:, 0].max():.3f}")
    print(f"  y: {xyz[:, 1].min():.3f} to {xyz[:, 1].max():.3f}")
    print(f"  z: {xyz[:, 2].min():.3f} to {xyz[:, 2].max():.3f}")

    print("\nIntensity range:")
    print(f"  intensity: {intensity.min():.3f} to {intensity.max():.3f}")

    print("\nFirst 5 points:")
    print(points[:5])


if __name__ == "__main__":
    main()
