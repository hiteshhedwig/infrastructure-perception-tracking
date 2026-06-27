import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_pcd_ascii(path: Path):
    with open(path, "r") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DATA"):
            data_start = i + 1
            break

    if data_start is None:
        raise RuntimeError(f"No DATA line found in {path}")

    pts = np.loadtxt(lines[data_start:], dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    return pts


def quat_to_yaw(qx, qy, qz, qw):
    return 2.0 * math.atan2(qz, qw)


def parse_openlabel_boxes(path: Path, score_thr=0.0):
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    openlabel = data.get("openlabel", data)
    frames = openlabel.get("frames", {})

    boxes = []

    for _, frame in frames.items():
        objects = frame.get("objects", {})

        for obj_id, obj in objects.items():
            od = obj.get("object_data", {})
            cls = od.get("type", "UNKNOWN")

            cuboid = od.get("cuboid", {})
            val = cuboid.get("val", None)
            if val is None or len(val) < 10:
                continue

            x, y, z, qx, qy, qz, qw, l, w, h = val[:10]

            score = 1.0
            attrs = cuboid.get("attributes", {})
            for item in attrs.get("num", []):
                if item.get("name") == "score":
                    score = float(item.get("val"))

            if score < score_thr:
                continue

            boxes.append({
                "id": obj_id,
                "class_name": cls,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "l": float(l),
                "w": float(w),
                "h": float(h),
                "yaw": float(quat_to_yaw(qx, qy, qz, qw)),
                "score": float(score),
            })

    return boxes


def box_corners_3d(box):
    x, y, z = box["x"], box["y"], box["z"]
    l, w, h = box["l"], box["w"], box["h"]
    yaw = box["yaw"]

    # bottom rectangle is enough for BEV, but create 8 corners for transform robustness
    xs = [ l / 2,  l / 2, -l / 2, -l / 2,  l / 2,  l / 2, -l / 2, -l / 2]
    ys = [ w / 2, -w / 2, -w / 2,  w / 2,  w / 2, -w / 2, -w / 2,  w / 2]
    zs = [-h / 2, -h / 2, -h / 2, -h / 2,  h / 2,  h / 2,  h / 2,  h / 2]

    pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float32)

    pts = pts @ R.T
    pts += np.array([x, y, z], dtype=np.float32)
    return pts


def bev_corners_2d(box):
    corners = box_corners_3d(box)
    return corners[:4, :2]


def transform_points(points_xyz, T):
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_xyz, ones], axis=1)
    out = homo @ T.T
    return out[:, :3]


def maybe_matrix_from_list(v):
    if not isinstance(v, list):
        return None

    # flat 16
    if len(v) == 16 and all(isinstance(x, (int, float)) for x in v):
        return np.array(v, dtype=np.float32).reshape(4, 4)

    # nested 4x4
    if len(v) == 4 and all(isinstance(row, list) and len(row) == 4 for row in v):
        flat = [x for row in v for x in row]
        if all(isinstance(x, (int, float)) for x in flat):
            return np.array(v, dtype=np.float32)

    return None


def find_matrices(obj, path="root"):
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            mat = maybe_matrix_from_list(v)
            if mat is not None:
                found.append((f"{path}.{k}", mat))

            found.extend(find_matrices(v, f"{path}.{k}"))

    elif isinstance(obj, list):
        mat = maybe_matrix_from_list(obj)
        if mat is not None:
            found.append((path, mat))
        else:
            for i, v in enumerate(obj):
                found.extend(find_matrices(v, f"{path}[{i}]"))

    return found


def world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H):
    px = int((x - x_min) / (x_max - x_min) * (W - 1))
    py = int((y_max - y) / (y_max - y_min) * (H - 1))
    return px, py


def draw_scene(points, gt_boxes, pred_boxes, title, out_path, x_min, x_max, y_min, y_max, transform=None):
    W, H = 1300, 950
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)

    # grid
    for gx in range(int(math.floor(x_min / 20) * 20), int(x_max) + 1, 20):
        p1 = world_to_pixel(gx, y_min, x_min, x_max, y_min, y_max, W, H)
        p2 = world_to_pixel(gx, y_max, x_min, x_max, y_min, y_max, W, H)
        cv2.line(img, p1, p2, (45, 45, 45), 1)

    for gy in range(int(math.floor(y_min / 20) * 20), int(y_max) + 1, 20):
        p1 = world_to_pixel(x_min, gy, x_min, x_max, y_min, y_max, W, H)
        p2 = world_to_pixel(x_max, gy, x_min, x_max, y_min, y_max, W, H)
        cv2.line(img, p1, p2, (45, 45, 45), 1)

    # points
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    pts = points[mask]
    if len(pts) > 0:
        if len(pts) > 120000:
            pts = pts[::2]

        z = pts[:, 2]
        z_norm = np.clip((z - (-8.0)) / 12.0, 0, 1)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]

        px = ((pts[:, 0] - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
        py = ((y_max - pts[:, 1]) / (y_max - y_min) * (H - 1)).astype(np.int32)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        img[py[valid], px[valid]] = colors[valid]

    # GT boxes green
    for box in gt_boxes:
        corners = bev_corners_2d(box)
        pts_px = np.array([
            world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H)
            for x, y in corners
        ], dtype=np.int32)
        cv2.polylines(img, [pts_px], True, (80, 255, 80), 3)

        cx, cy = world_to_pixel(box["x"], box["y"], x_min, x_max, y_min, y_max, W, H)
        cv2.putText(img, f"GT {box['class_name']}", (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1, cv2.LINE_AA)

    # Pred boxes red/yellow
    for box in pred_boxes:
        corners3d = box_corners_3d(box)

        if transform is not None:
            corners3d = transform_points(corners3d, transform)
            color = (0, 220, 255)
            label_prefix = "TX"
        else:
            color = (60, 60, 255)
            label_prefix = "P"

        corners2d = corners3d[:4, :2]
        pts_px = np.array([
            world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H)
            for x, y in corners2d
        ], dtype=np.int32)
        cv2.polylines(img, [pts_px], True, color, 3)

        center = corners3d.mean(axis=0)
        cx, cy = world_to_pixel(center[0], center[1], x_min, x_max, y_min, y_max, W, H)
        cv2.putText(img, f"{label_prefix} {box['class_name']} {box['score']:.2f}", (cx + 4, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (W, 72), (0, 0, 0), -1)
    cv2.putText(img, title[:130], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "green = GT | red = raw pred | yellow = transformed pred candidate", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset")
    parser.add_argument("--pred-root", default="external/coopdet3d/inference_train_openlabel_pretrained/openlabel")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-idx", type=int, default=153)
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--out-dir", default="outputs/debug_pred_transform_candidates")
    parser.add_argument("--x-min", type=float, default=-80)
    parser.add_argument("--x-max", type=float, default=120)
    parser.add_argument("--y-min", type=float, default=-100)
    parser.add_argument("--y-max", type=float, default=120)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = dataset_root / args.split / "labels_point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
    pcd_dir = dataset_root / args.split / "point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"

    gt_files = sorted(gt_dir.glob("*.json"))
    gt_path = gt_files[args.sample_idx]
    frame_id = gt_path.stem
    pred_path = pred_root / f"{frame_id}.json"
    pcd_path = pcd_dir / f"{frame_id}.pcd"

    print("frame idx:", args.sample_idx)
    print("frame id:", frame_id)
    print("gt path:", gt_path)
    print("pred path:", pred_path, "exists:", pred_path.exists())
    print("pcd path:", pcd_path)

    points = load_pcd_ascii(pcd_path)
    gt_boxes = parse_openlabel_boxes(gt_path, score_thr=0.0)
    pred_boxes = parse_openlabel_boxes(pred_path, score_thr=args.score_thr)

    print("GT boxes:", len(gt_boxes))
    print("Pred boxes:", len(pred_boxes))

    with open(gt_path, "r") as f:
        gt_json = json.load(f)

    matrices = find_matrices(gt_json)
    unique = []
    seen = set()

    for name, M in matrices:
        key = tuple(np.round(M.flatten(), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, M))

    print("\nFound matrices:")
    for i, (name, M) in enumerate(unique):
        print(f"[{i}] {name}")
        print(M)

    # raw
    draw_scene(
        points, gt_boxes, pred_boxes,
        title=f"{args.sample_idx} {frame_id} | RAW predictions",
        out_path=out_dir / f"{args.sample_idx:04d}_00_raw_pred.jpg",
        x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max,
        transform=None,
    )

    # candidates
    for i, (name, M) in enumerate(unique[:20]):
        draw_scene(
            points, gt_boxes, pred_boxes,
            title=f"{args.sample_idx} | FORWARD matrix {i}: {name}",
            out_path=out_dir / f"{args.sample_idx:04d}_{i+1:02d}_forward.jpg",
            x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max,
            transform=M,
        )

        try:
            Minv = np.linalg.inv(M)
            draw_scene(
                points, gt_boxes, pred_boxes,
                title=f"{args.sample_idx} | INVERSE matrix {i}: {name}",
                out_path=out_dir / f"{args.sample_idx:04d}_{i+1:02d}_inverse.jpg",
                x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max,
                transform=Minv,
            )
        except Exception as e:
            print("Could not invert matrix", i, name, e)

    print("\nWrote:", out_dir)


if __name__ == "__main__":
    main()
