import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


CLASS_ORDER = [
    "CAR", "TRUCK", "TRAILER", "VAN", "BUS",
    "PEDESTRIAN", "BICYCLE", "MOTORCYCLE", "EMERGENCY_VEHICLE"
]


def load_pcd_ascii(path: Path):
    with open(path, "r") as f:
        lines = f.readlines()

    data_start = None
    fields = None

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        if line.startswith("DATA"):
            data_start = i + 1
            break

    if data_start is None:
        raise RuntimeError(f"No DATA line found in {path}")

    data = np.loadtxt(lines[data_start:], dtype=np.float32)

    if data.ndim == 1:
        data = data[None, :]

    return data, fields


def quat_to_yaw(qx, qy, qz, qw):
    return 2.0 * math.atan2(qz, qw)


def get_attr(attrs, attr_type, name, default=None):
    for item in attrs.get(attr_type, []):
        if item.get("name") == name:
            return item.get("val", default)
    return default


def parse_openlabel_boxes(path: Path, is_pred=False):
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)


    # Raw CoopDet3D prediction format exported directly from outputs[0]["boxes_3d"].
    # This bypasses the broken OpenLABEL conversion.
    if data.get("format") == "coopdet3d_raw_lidar_boxes_v1":
        boxes = []
        for b in data.get("boxes", []):
            score = float(b.get("score", 1.0))
            boxes.append({
                "id": "",
                "class_name": str(b["class_name"]).upper(),
                "center": np.array(
                    [float(b["x"]), float(b["y"]), float(b["z"])],
                    dtype=np.float32,
                ),
                "dims_lwh": np.array(
                    [float(b["l"]), float(b["w"]), float(b["h"])],
                    dtype=np.float32,
                ),
                "yaw": float(b["yaw"]),
                "score": score,
                "num_points": None,
                "is_pred": is_pred,
            })
        return boxes

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
            attrs = cuboid.get("attributes", {})

            score = get_attr(attrs, "num", "score", None)
            num_points = get_attr(attrs, "num", "num_points", None)

            boxes.append({
                "id": obj_id,
                "class_name": cls,
                "center": np.array([x, y, z], dtype=np.float32),
                "dims_lwh": np.array([l, w, h], dtype=np.float32),
                "yaw": quat_to_yaw(qx, qy, qz, qw),
                "score": score,
                "num_points": num_points,
                "is_pred": is_pred,
            })

    return boxes


def bev_corners(box):
    x, y, _ = box["center"]
    l, w, _ = box["dims_lwh"]
    yaw = box["yaw"]

    local = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
    ], dtype=np.float32)

    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    return local @ R.T + np.array([x, y], dtype=np.float32)


def polygon_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0
    poly = np.asarray(poly, dtype=np.float32)
    return float(abs(cv2.contourArea(poly)))


def bev_iou(box_a, box_b):
    pa = bev_corners(box_a).astype(np.float32)
    pb = bev_corners(box_b).astype(np.float32)

    area_a = polygon_area(pa)
    area_b = polygon_area(pb)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    ret, inter = cv2.intersectConvexConvex(pa, pb)
    inter_area = float(ret)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def greedy_match(gt_boxes, pred_boxes, iou_thr=0.1, same_class=True):
    candidates = []

    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            if same_class and g["class_name"] != p["class_name"]:
                continue

            iou = bev_iou(g, p)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    return matches, matched_gt, matched_pred


def world_to_pixel_xy(x, y, x_min, x_max, y_min, y_max, W, H):
    px = int((x - x_min) / (x_max - x_min) * (W - 1))
    py = int((y_max - y) / (y_max - y_min) * (H - 1))
    return px, py


def draw_panel(points, gt_boxes, pred_boxes, matches, matched_gt, matched_pred,
               title, x_min, x_max, y_min, y_max, W=1100, H=900):

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)

    # Grid
    for gx in range(int(math.floor(x_min / 20) * 20), int(x_max) + 1, 20):
        p1 = world_to_pixel_xy(gx, y_min, x_min, x_max, y_min, y_max, W, H)
        p2 = world_to_pixel_xy(gx, y_max, x_min, x_max, y_min, y_max, W, H)
        cv2.line(img, p1, p2, (45, 45, 45), 1)

    for gy in range(int(math.floor(y_min / 20) * 20), int(y_max) + 1, 20):
        p1 = world_to_pixel_xy(x_min, gy, x_min, x_max, y_min, y_max, W, H)
        p2 = world_to_pixel_xy(x_max, gy, x_min, x_max, y_min, y_max, W, H)
        cv2.line(img, p1, p2, (45, 45, 45), 1)

    # Points
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    pts = points[mask]

    if len(pts) > 0:
        z = pts[:, 2]
        z_norm = np.clip((z - (-8.0)) / 12.0, 0, 1)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]

        px = ((pts[:, 0] - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
        py = ((y_max - pts[:, 1]) / (y_max - y_min) * (H - 1)).astype(np.int32)

        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        img[py[valid], px[valid]] = colors[valid]

    gt_color = (80, 255, 80)
    pred_color = (60, 80, 255)
    match_color = (0, 220, 255)

    match_lookup_gt = {gi: (pi, iou) for gi, pi, iou in matches}
    match_lookup_pred = {pi: (gi, iou) for gi, pi, iou in matches}

    # GT boxes
    for gi, box in enumerate(gt_boxes):
        corners = bev_corners(box)
        pts_px = np.array([
            world_to_pixel_xy(x, y, x_min, x_max, y_min, y_max, W, H)
            for x, y in corners
        ], dtype=np.int32)

        color = match_color if gi in matched_gt else gt_color
        thickness = 2 if gi in matched_gt else 3

        cv2.polylines(img, [pts_px], True, color, thickness)

        cx, cy = world_to_pixel_xy(box["center"][0], box["center"][1], x_min, x_max, y_min, y_max, W, H)
        label = f"GT {box['class_name']}"
        if gi in match_lookup_gt:
            label += f" IoU:{match_lookup_gt[gi][1]:.2f}"
        cv2.putText(img, label, (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Pred boxes
    for pi, box in enumerate(pred_boxes):
        corners = bev_corners(box)
        pts_px = np.array([
            world_to_pixel_xy(x, y, x_min, x_max, y_min, y_max, W, H)
            for x, y in corners
        ], dtype=np.int32)

        color = match_color if pi in matched_pred else pred_color
        thickness = 2 if pi in matched_pred else 3

        cv2.polylines(img, [pts_px], True, color, thickness)

        cx, cy = world_to_pixel_xy(box["center"][0], box["center"][1], x_min, x_max, y_min, y_max, W, H)
        score = box.get("score", None)
        score_txt = f" {score:.2f}" if isinstance(score, (int, float)) else ""
        label = f"P {box['class_name']}{score_txt}"
        cv2.putText(img, label, (cx + 3, cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    tp = len(matches)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    cv2.rectangle(img, (0, 0), (W, 80), (0, 0, 0), -1)
    cv2.putText(img, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"GT: {len(gt_boxes)} | Pred: {len(pred_boxes)} | Match: {tp} | FP: {fp} | FN: {fn}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(img, "GT green | Pred red/blue | Matched yellow", (W - 430, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset")
    parser.add_argument("--pred-root", default="external/coopdet3d/inference_val_openlabel/openlabel")
    parser.add_argument("--out-dir", default="outputs/coopdet3d_gt_pred_bev_val")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-indices", default="0,1,2,3,4,5,10,20,50,99")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.1)
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

    if not gt_files:
        raise RuntimeError(f"No GT json files found in {gt_dir}")

    if args.sample_indices == "all":
        indices = list(range(len(gt_files)))
    else:
        indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]

    summary_rows = []

    for idx in indices:
        gt_path = gt_files[idx]
        frame_id = gt_path.stem
        pcd_path = pcd_dir / f"{frame_id}.pcd"
        pred_path = pred_root / f"{frame_id}.json"

        if not pcd_path.exists():
            print(f"SKIP missing PCD: {pcd_path}")
            continue

        points, _ = load_pcd_ascii(pcd_path)

        gt_boxes = parse_openlabel_boxes(gt_path, is_pred=False)
        pred_boxes = parse_openlabel_boxes(pred_path, is_pred=True)

        pred_boxes = [
            b for b in pred_boxes
            if b.get("score") is None or float(b["score"]) >= args.score_thr
        ]

        matches, matched_gt, matched_pred = greedy_match(
            gt_boxes,
            pred_boxes,
            iou_thr=args.iou_thr,
            same_class=True,
        )

        img = draw_panel(
            points,
            gt_boxes,
            pred_boxes,
            matches,
            matched_gt,
            matched_pred,
            title=f"{idx:03d} | {frame_id}",
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )

        out_path = out_dir / f"{idx:03d}_{frame_id}_gt_pred_bev.jpg"
        cv2.imwrite(str(out_path), img)

        tp = len(matches)
        fp = len(pred_boxes) - len(matched_pred)
        fn = len(gt_boxes) - len(matched_gt)

        mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        summary_rows.append((idx, frame_id, len(gt_boxes), len(pred_boxes), tp, fp, fn, mean_iou, pred_path.exists()))

        print(f"WROTE {out_path} | GT={len(gt_boxes)} Pred={len(pred_boxes)} TP={tp} FP={fp} FN={fn} meanIoU={mean_iou:.3f} pred_file={pred_path.exists()}")

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("sample_idx,frame_id,gt,pred,tp,fp,fn,mean_iou,pred_file_exists\n")
        for row in summary_rows:
            f.write(",".join(map(str, row)) + "\n")

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
