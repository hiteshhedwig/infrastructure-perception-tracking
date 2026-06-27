#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


DEFAULT_FRAME_ID = "1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
DEFAULT_SENSOR = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"


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


def get_first_cuboid(object_data):
    cuboid = object_data.get("cuboid")

    if cuboid is None:
        return None

    if isinstance(cuboid, list):
        if len(cuboid) == 0:
            return None
        return cuboid[0]

    return cuboid


def parse_label(label_path: Path, camera_id: str):
    with open(label_path, "r") as f:
        data = json.load(f)

    frame = next(iter(data["openlabel"]["frames"].values()))
    frame_properties = frame.get("frame_properties", {})

    image_file_names = frame_properties.get("image_file_names", [])
    image_file_name = None

    for name in image_file_names:
        if camera_id in name:
            image_file_name = name
            break

    if image_file_name is None:
        raise ValueError(f"No image found for camera_id={camera_id}")

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

        boxes.append(
            {
                "track_id": object_id,
                "class_name": class_name,
                "center": np.array([x, y, z], dtype=np.float64),
                "quat": np.array([qx, qy, qz, qw], dtype=np.float64),
                "size": np.array([length, width, height], dtype=np.float64),
            }
        )

    return boxes, image_file_name


def box_3d_corners(box):
    l, w, h = box["size"]

    local = np.array(
        [
            [ l / 2,  w / 2, -h / 2],
            [ l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, -h / 2],
            [-l / 2,  w / 2, -h / 2],
            [ l / 2,  w / 2,  h / 2],
            [ l / 2, -w / 2,  h / 2],
            [-l / 2, -w / 2,  h / 2],
            [-l / 2,  w / 2,  h / 2],
        ],
        dtype=np.float64,
    )

    rot = R.from_quat(box["quat"]).as_matrix()
    corners = local @ rot.T
    corners += box["center"]

    return corners


def project_xyz(points_xyz: np.ndarray, projection_matrix: np.ndarray):
    points_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
    proj = points_h @ projection_matrix.T

    depth = proj[:, 2]
    valid_depth = depth > 1e-6

    uv = np.zeros((points_xyz.shape[0], 2), dtype=np.float64)
    uv[valid_depth, 0] = proj[valid_depth, 0] / depth[valid_depth]
    uv[valid_depth, 1] = proj[valid_depth, 1] / depth[valid_depth]

    return uv, depth, valid_depth


def draw_projected_points(img, points, projection_matrix, max_points=30000):
    xyz = points[:, :3]

    # Keep useful vertical range only.
    z = xyz[:, 2]
    mask = (z > -9.0) & (z < 3.0)
    xyz = xyz[mask]

    if xyz.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]

    uv, depth, valid_depth = project_xyz(xyz, projection_matrix)

    h, w = img.shape[:2]

    valid = (
        valid_depth
        & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    uv_valid = uv[valid].astype(np.int32)
    depth_valid = depth[valid]

    print(f"Projected LiDAR points inside image: {len(uv_valid)}")

    if len(uv_valid) == 0:
        return img

    d_min, d_max = np.percentile(depth_valid, [2, 98])
    depth_norm = np.clip((depth_valid - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    colors = cv2.applyColorMap((255 * (1 - depth_norm)).astype(np.uint8), cv2.COLORMAP_TURBO)

    for (u, v), color in zip(uv_valid, colors):
        img[v, u] = color[0].tolist()

    return img


def draw_projected_boxes(img, boxes, projection_matrix):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    h, w = img.shape[:2]
    drawn = 0

    for box in boxes:
        corners = box_3d_corners(box)
        uv, depth, valid_depth = project_xyz(corners, projection_matrix)

        # Need at least a few visible corners.
        inside = (
            valid_depth
            & (uv[:, 0] >= -200) & (uv[:, 0] < w + 200)
            & (uv[:, 1] >= -200) & (uv[:, 1] < h + 200)
        )

        if inside.sum() < 2:
            continue

        uv_int = uv.astype(np.int32)

        color = (0, 255, 255)

        for a, b in edges:
            if valid_depth[a] and valid_depth[b]:
                pt1 = tuple(uv_int[a])
                pt2 = tuple(uv_int[b])
                cv2.line(img, pt1, pt2, color, 2, lineType=cv2.LINE_AA)

        center_uv, center_depth, center_valid = project_xyz(
            box["center"].reshape(1, 3),
            projection_matrix,
        )

        if center_valid[0]:
            u, v = center_uv[0].astype(int)
            if -100 <= u < w + 100 and -100 <= v < h + 100:
                label = f"{box['class_name']}_{box['track_id'][:4]}"
                cv2.putText(
                    img,
                    label,
                    (u, v),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        drawn += 1

    print(f"Projected boxes drawn: {drawn}/{len(boxes)}")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/tumtraf_v2x")
    parser.add_argument("--split", default="train")
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--sensor", default=DEFAULT_SENSOR)
    parser.add_argument("--camera-id", default="s110_camera_basler_south1_8mm")
    parser.add_argument("--calib-path", default="external/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_south1_8mm_R4.json")
    parser.add_argument("--draw-points", action="store_true")
    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / args.split

    label_path = split_root / "labels_point_clouds" / args.sensor / f"{args.frame_id}.json"
    pcd_path = split_root / "point_clouds" / args.sensor / f"{args.frame_id}.pcd"

    boxes, image_file_name = parse_label(label_path, args.camera_id)
    image_path = split_root / "images" / args.camera_id / image_file_name

    with open(args.calib_path, "r") as f:
        calib = json.load(f)

    projection_matrix = np.array(calib["projection_from_s110_lidar_ouster_south"], dtype=np.float64)

    if projection_matrix.shape != (3, 4):
        raise ValueError(f"Bad projection matrix shape: {projection_matrix.shape}")

    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"PCD:   {pcd_path}")
    print(f"Calib: {args.calib_path}")
    print(f"Boxes parsed: {len(boxes)}")
    print(f"Projection matrix shape: {projection_matrix.shape}")

    if args.draw_points:
        points = load_ascii_pcd(pcd_path)
        img = draw_projected_points(img, points, projection_matrix)

    if args.draw_boxes:
        img = draw_projected_boxes(img, boxes, projection_matrix)

    if args.output is None:
        output_path = Path("outputs/projections/r4_direct") / f"{args.frame_id}_{args.camera_id}.jpg"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
