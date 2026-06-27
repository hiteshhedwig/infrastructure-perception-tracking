#!/usr/bin/env python3

from pathlib import Path
import json
import cv2
import numpy as np


DATA_ROOT = Path("data/raw/tumtraf_v2x")
SPLIT = "train"
REGISTERED_SENSOR = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
SOUTH_LIDAR_SENSOR = "s110_lidar_ouster_south"
FRAME_ID = "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"

CALIB_ROOT = Path("external/tum-traffic-dataset-dev-kit/calib")
OUTPUT_DIR = Path("outputs/projections/south_lidar_only")


def load_ascii_pcd(pcd_path: Path) -> np.ndarray:
    data_start_line = None

    with open(pcd_path, "r") as f:
        for line_idx, line in enumerate(f):
            if line.strip().startswith("DATA"):
                data_start_line = line_idx + 1
                break

    if data_start_line is None:
        raise ValueError(f"Could not find DATA line in {pcd_path}")

    return np.loadtxt(pcd_path, skiprows=data_start_line)


def parse_frame_properties(label_path: Path):
    with open(label_path, "r") as f:
        data = json.load(f)

    frame = next(iter(data["openlabel"]["frames"].values()))
    return frame["frame_properties"]


def camera_id_from_image_name(image_name: str):
    stem = Path(image_name).stem
    parts = stem.split("_")
    return "_".join(parts[2:])


def project_points(points_xyz: np.ndarray, projection_matrix: np.ndarray):
    points_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
    projected = points_h @ projection_matrix.T

    depth = projected[:, 2]
    valid_depth = depth > 1e-6

    uv = np.zeros((points_xyz.shape[0], 2), dtype=np.float64)
    uv[valid_depth, 0] = projected[valid_depth, 0] / depth[valid_depth]
    uv[valid_depth, 1] = projected[valid_depth, 1] / depth[valid_depth]

    return uv, depth, valid_depth


def draw_points(image, points, projection_matrix, max_points=60000):
    xyz = points[:, :3]

    # Keep road/object useful height.
    z = xyz[:, 2]
    mask = (z > -9.0) & (z < 3.0)
    xyz = xyz[mask]

    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]

    uv, depth, valid_depth = project_points(xyz, projection_matrix)

    h, w = image.shape[:2]

    valid = (
        valid_depth
        & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    uv_valid = uv[valid].astype(np.int32)
    depth_valid = depth[valid]

    print(f"    points inside image: {len(uv_valid)}")

    if len(uv_valid) == 0:
        return image

    d_min, d_max = np.percentile(depth_valid, [2, 98])
    depth_norm = np.clip((depth_valid - d_min) / max(d_max - d_min, 1e-6), 0, 1)

    colors = cv2.applyColorMap(
        (255 * (1 - depth_norm)).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )

    overlay = image.copy()

    for (u, v), color in zip(uv_valid, colors):
        cv2.circle(overlay, (u, v), 1, color[0].tolist(), -1)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_root = DATA_ROOT / SPLIT

    label_path = (
        split_root
        / "labels_point_clouds"
        / REGISTERED_SENSOR
        / f"{FRAME_ID}.json"
    )

    frame_props = parse_frame_properties(label_path)

    south_pcd_name = None
    for name in frame_props["point_cloud_file_names"]:
        if name.endswith("_s110_lidar_ouster_south.pcd"):
            south_pcd_name = name
            break

    if south_pcd_name is None:
        raise ValueError("Could not find raw s110_lidar_ouster_south PCD in frame_properties.")

    south_pcd_path = split_root / "point_clouds" / SOUTH_LIDAR_SENSOR / south_pcd_name

    print(f"Label:      {label_path}")
    print(f"South PCD:  {south_pcd_path}")

    points = load_ascii_pcd(south_pcd_path)
    print(f"Loaded south-only points: {points.shape}")

    for image_name in frame_props["image_file_names"]:
        camera_id = camera_id_from_image_name(image_name)
        image_path = split_root / "images" / camera_id / image_name

        calib_candidates = [
            CALIB_ROOT / f"{camera_id}_R4.json",
            CALIB_ROOT / f"{camera_id}.json",
        ]

        print()
        print(f"Camera: {camera_id}")
        print(f"  image: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            print("  could not read image")
            continue

        for calib_path in calib_candidates:
            if not calib_path.exists():
                continue

            with open(calib_path, "r") as f:
                calib = json.load(f)

            projection = calib.get("projection_from_s110_lidar_ouster_south", None)

            if not projection:
                print(f"  no projection_from_s110_lidar_ouster_south in {calib_path.name}")
                continue

            projection = np.array(projection, dtype=np.float64)

            if projection.shape != (3, 4):
                print(f"  bad projection shape: {projection.shape}")
                continue

            print(f"  calib: {calib_path.name}")

            out = draw_points(image.copy(), points, projection)

            output_path = OUTPUT_DIR / f"{camera_id}_{calib_path.stem}_south_lidar_only.jpg"
            cv2.imwrite(str(output_path), out)

            print(f"    saved: {output_path}")


if __name__ == "__main__":
    main()
