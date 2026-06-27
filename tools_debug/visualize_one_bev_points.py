#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PCD_PATH = Path(
    "data/raw/tumtraf_v2x/train/point_clouds/"
    "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/"
    "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.pcd"
)

OUTPUT_PATH = Path("outputs/bev/one_frame_pointcloud_bev.png")


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

    points = np.loadtxt(pcd_path, skiprows=data_start_line)
    return points


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {PCD_PATH}")
    points = load_ascii_pcd(PCD_PATH)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    print(f"Loaded points: {points.shape[0]}")

    # Basic crop to remove very high/low noise.
    # We are not choosing model limits yet; this is only for visualization.
    mask = (z > -9.0) & (z < 2.0)

    x_crop = x[mask]
    y_crop = y[mask]
    intensity_crop = intensity[mask]

    print(f"Points after z crop: {x_crop.shape[0]}")

    plt.figure(figsize=(10, 10))

    plt.scatter(
        x_crop,
        y_crop,
        c=intensity_crop,
        s=0.2,
        cmap="gray",
        marker="."
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("TUMTraf V2X - Registered Point Cloud BEV")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved BEV image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
