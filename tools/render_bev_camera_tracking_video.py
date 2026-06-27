import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from mmcv import Config
from torchpack.utils.config import configs

# Make CoopDet3D imports work when this script is run from /home/cpu-57/heetez/ipt
COOP_ROOT = Path("external/coopdet3d").resolve()
sys.path.insert(0, str(COOP_ROOT))

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def replace_val_with_train(obj):
    if isinstance(obj, dict):
        return {k: replace_val_with_train(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_val_with_train(v) for v in obj]
    if isinstance(obj, str):
        return (
            obj.replace("tumtraf_v2x_nusc_infos_val.pkl", "tumtraf_v2x_nusc_infos_train.pkl")
               .replace("/val/", "/train/")
               .replace("/validation/", "/training/")
        )
    return obj


def track_color(track_id):
    tid = int(track_id)
    rng = np.random.default_rng(tid * 7919)
    color = rng.integers(60, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def make_lidar_boxes(rows):
    arr = []
    ids = []
    labels = []

    for _, r in rows.iterrows():
        # MMDet3D LiDAR box format: x, y, z, dx, dy, dz, yaw, vx, vy
        x = float(r["x"])
        y = float(r["y"])
        z = float(r["z"])
        l = float(r["l"])
        w = float(r["w"])
        h = float(r["h"])
        yaw = float(r["yaw"])

        arr.append([x, y, z, l, w, h, yaw, 0.0, 0.0])
        ids.append(int(r["track_id"]))
        labels.append(str(r.get("class_name", "OBJ")))

    if not arr:
        return None, [], []

    arr = np.asarray(arr, dtype=np.float32)
    boxes = LiDARInstance3DBoxes(arr, box_dim=9)
    return boxes, ids, labels


def project_points(points_xyz, transform):
    pts = np.asarray(points_xyz, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)

    T = np.asarray(transform, dtype=np.float32)
    proj = pts_h @ T.T

    depth = proj[:, 2]
    uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
    valid = depth > 1e-3
    uv[valid, 0] = proj[valid, 0] / depth[valid]
    uv[valid, 1] = proj[valid, 1] / depth[valid]
    return uv, valid


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def draw_projected_tracks(image, rows, transform, title):
    img = image.copy()

    boxes, track_ids, labels = make_lidar_boxes(rows)
    if boxes is None:
        cv2.putText(img, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    corners = boxes.corners.detach().cpu().numpy()  # [N, 8, 3]

    H, W = img.shape[:2]

    for bi in range(corners.shape[0]):
        uv, valid = project_points(corners[bi], transform)

        # Skip boxes completely behind camera.
        if valid.sum() < 4:
            continue

        tid = track_ids[bi]
        cls = labels[bi]
        color = track_color(tid)

        pts = uv.astype(np.int32)

        # Draw if projected points are not wildly out of view.
        if (
            np.nanmax(pts[:, 0]) < -W or np.nanmin(pts[:, 0]) > 2 * W or
            np.nanmax(pts[:, 1]) < -H or np.nanmin(pts[:, 1]) > 2 * H
        ):
            continue

        for a, b in BOX_EDGES:
            if valid[a] and valid[b]:
                pa = tuple(pts[a])
                pb = tuple(pts[b])

                # Black underlay + colored line for visibility.
                cv2.line(img, pa, pb, (0, 0, 0), 5)
                cv2.line(img, pa, pb, color, 3)

        # Label near projected center.
        center = corners[bi].mean(axis=0, keepdims=True)
        center_uv, center_valid = project_points(center, transform)
        if center_valid[0]:
            cx, cy = center_uv[0].astype(int)
            if -100 <= cx <= W + 100 and -100 <= cy <= H + 100:
                label = f"T{tid}"

                # Larger readable label: black outline + white fill.
                cv2.putText(
                    img,
                    label,
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    label,
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    cv2.rectangle(img, (0, 0), (W, 42), (0, 0, 0), -1)
    cv2.putText(img, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def resize_keep(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--bev-frame-dir", required=True)
    parser.add_argument("--tracking-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--train-as-eval", action="store_true")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args, opts = parser.parse_known_args()

    # Resolve paths from the project root before changing cwd.
    config_path = Path(args.config).resolve()
    bev_frame_dir = Path(args.bev_frame_dir).resolve()
    tracking_csv = Path(args.tracking_csv).resolve()
    out_dir = Path(args.out_dir).resolve()

    # CoopDet3D configs use relative paths like data/...
    # They must be resolved from the CoopDet3D repo root.
    os.chdir(COOP_ROOT)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_out = out_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    configs.load(str(config_path), recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=str(config_path))

    if args.train_as_eval:
        dataset_cfg = copy.deepcopy(cfg.data.val)
        dataset_cfg = replace_val_with_train(dataset_cfg)
        effective_split = "train"
    else:
        dataset_cfg = copy.deepcopy(cfg.data[args.split])
        effective_split = args.split

    dataset = build_dataset(dataset_cfg)
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )

    df = pd.read_csv(tracking_csv)

    if "sample_idx" in df.columns:
        sample_col = "sample_idx"
    elif "frame_idx" in df.columns:
        sample_col = "frame_idx"
    else:
        raise RuntimeError(
            f"diagnostics.csv must contain sample_idx or frame_idx. Columns: {list(df.columns)}"
        )

    required_cols = ["track_id", "x", "y", "z", "l", "w", "h", "yaw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"diagnostics.csv missing columns: {missing}. Columns: {list(df.columns)}")

    bev_paths = sorted(
        list(bev_frame_dir.glob("*.jpg")) +
        list(bev_frame_dir.glob("*.png"))
    )

    if len(bev_paths) < args.max_frames:
        raise RuntimeError(f"Not enough BEV frames in {bev_frame_dir}: {len(bev_paths)} < {args.max_frames}")

    frame_w, frame_h = 1920, 1080
    bev_w, bev_h = 1150, 1080
    cam_grid_w, cam_grid_h = frame_w - bev_w, frame_h
    cam_w, cam_h = cam_grid_w // 2, cam_grid_h // 2

    video_path = out_dir / "bev_plus_cameras.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (frame_w, frame_h),
    )

    selected_data = []
    for idx, data in enumerate(dataflow):
        if idx < args.start_index:
            continue
        if len(selected_data) >= args.max_frames:
            break
        selected_data.append((idx, data))

    print("Selected frames:", len(selected_data))

    for local_i, (sample_idx, data) in enumerate(selected_data):
        metas = data["metas"].data[0][0]

        rows = df[df[sample_col] == sample_idx].copy()

        if "time_since_update" in rows.columns:
            rows = rows[rows["time_since_update"] == 0]

        bev = cv2.imread(str(bev_paths[local_i]))
        if bev is None:
            raise RuntimeError(f"Could not read BEV frame: {bev_paths[local_i]}")

        bev_panel = resize_keep(bev, bev_w, bev_h)

        cam_imgs = []
        cam_titles = []

        # Infrastructure cameras.
        for k, img_path in enumerate(metas.get("infrastructure_filename", [])):
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                img[:] = (30, 30, 30)

            transform = metas["infrastructure_lidar2image"][k]
            img = draw_projected_tracks(img, rows, transform, f"Infra Cam {k}")
            cam_imgs.append(resize_keep(img, cam_w, cam_h))
            cam_titles.append(f"infra-{k}")

        # Vehicle camera.
        if "vehicle_filename" in metas:
            v2i = data["vehicle2infrastructure"].data[0].numpy().astype(np.float32)
            v2i = np.squeeze(v2i)

            for k, img_path in enumerate(metas.get("vehicle_filename", [])):
                img = cv2.imread(str(img_path))
                if img is None:
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)
                    img[:] = (30, 30, 30)

                transform = metas["vehicle_lidar2image"][k] @ np.linalg.inv(v2i)
                img = draw_projected_tracks(img, rows, transform, f"Vehicle Cam {k}")
                cam_imgs.append(resize_keep(img, cam_w, cam_h))
                cam_titles.append(f"vehicle-{k}")

        # Ensure exactly 4 camera panels.
        while len(cam_imgs) < 4:
            blank = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            blank[:] = (20, 20, 20)
            cam_imgs.append(blank)

        cam_grid = np.zeros((cam_grid_h, cam_grid_w, 3), dtype=np.uint8)
        cam_grid[0:cam_h, 0:cam_w] = cam_imgs[0]
        cam_grid[0:cam_h, cam_w:cam_grid_w] = cam_imgs[1]
        cam_grid[cam_h:cam_grid_h, 0:cam_w] = cam_imgs[2]
        cam_grid[cam_h:cam_grid_h, cam_w:cam_grid_w] = cam_imgs[3]

        combined = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        combined[:, :bev_w] = bev_panel
        combined[:, bev_w:] = cam_grid

        frame_id = rows["frame_id"].iloc[0] if "frame_id" in rows.columns and len(rows) else str(sample_idx)
        cv2.rectangle(combined, (0, 0), (frame_w, 44), (0, 0, 0), -1)
        cv2.putText(
            combined,
            f"BEV + Camera Tracking | sample {sample_idx} | tracks {len(rows)} | {frame_id}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_frame = frames_out / f"{local_i:06d}.jpg"
        cv2.imwrite(str(out_frame), combined)
        writer.write(combined)

        print(f"{local_i:03d} sample={sample_idx} tracks={len(rows)}")

    writer.release()

    print("Done.")
    print("Video:", video_path)
    print("Frames:", frames_out)


if __name__ == "__main__":
    main()
