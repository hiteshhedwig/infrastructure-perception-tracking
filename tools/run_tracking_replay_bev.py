import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


VEHICLE_CLASSES = {"CAR", "TRUCK", "TRAILER", "VAN", "BUS", "EMERGENCY_VEHICLE"}
SMALL_CLASSES = {"PEDESTRIAN", "BICYCLE", "MOTORCYCLE"}


def parse_timestamp_from_frame_id(frame_id: str) -> float:
    # Example:
    # 1688625741_046595374_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered
    parts = frame_id.split("_")
    if len(parts) < 2:
        return 0.0
    try:
        sec = int(parts[0])
        nsec = int(parts[1])
        return sec + nsec * 1e-9
    except Exception:
        return 0.0


def load_pcd_ascii(path: Path) -> np.ndarray:
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


def get_attr(attrs, attr_type, name, default=None):
    for item in attrs.get(attr_type, []):
        if item.get("name") == name:
            return item.get("val", default)
    return default


def parse_openlabel_boxes(path: Path, score_thr=0.0):
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    # New trusted format: raw CoopDet3D boxes exported directly from
    # outputs[0]["boxes_3d"], with no OpenLABEL conversion.
    if data.get("format") == "coopdet3d_raw_lidar_boxes_v1":
        boxes = []
        for b in data.get("boxes", []):
            if float(b.get("score", 1.0)) < score_thr:
                continue
            boxes.append({
                "object_id": "",
                "class_name": str(b["class_name"]).upper(),
                "x": float(b["x"]),
                "y": float(b["y"]),
                "z": float(b["z"]),
                "l": float(b["l"]),
                "w": float(b["w"]),
                "h": float(b["h"]),
                "yaw": float(b["yaw"]),
                "score": float(b.get("score", 1.0)),
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
            if score is not None and float(score) < score_thr:
                continue

            boxes.append({
                "object_id": obj_id,
                "class_name": cls,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "l": float(l),
                "w": float(w),
                "h": float(h),
                "yaw": float(quat_to_yaw(qx, qy, qz, qw)),
                "score": float(score) if score is not None else 1.0,
            })

    return boxes


def detection_match_threshold(class_name: str) -> float:
    if class_name in SMALL_CLASSES:
        return 2.0
    if class_name in VEHICLE_CLASSES:
        return 4.0
    return 3.0


def classes_compatible(track_class: str, det_class: str) -> bool:
    # Exact class match is best.
    if track_class == det_class:
        return True

    # CoopDet3D sometimes flips between nearby vehicle categories across frames:
    # CAR/VAN/TRUCK/TRAILER/BUS. For tracking identity, those should still be
    # allowed to match. We keep pedestrians/bicycles/motorcycles stricter.
    if track_class in VEHICLE_CLASSES and det_class in VEHICLE_CLASSES:
        return True

    return False


class Track:
    def __init__(self, track_id, det, frame_idx):
        self.track_id = track_id
        self.class_name = det["class_name"]

        self.x = det["x"]
        self.y = det["y"]
        self.z = det["z"]

        self.vx = 0.0
        self.vy = 0.0

        self.l = det["l"]
        self.w = det["w"]
        self.h = det["h"]
        self.yaw = det["yaw"]
        self.score = det["score"]

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        self.history = [(self.x, self.y)]
        self.last_match_distance = None
        self.last_detection_score = self.score

    def predict(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += 1
        self.time_since_update += 1
        self.history.append((self.x, self.y))

    def update(self, det, frame_idx, dt, match_distance):
        old_x, old_y = self.x, self.y

        if dt > 1e-6:
            meas_vx = (det["x"] - old_x) / dt
            meas_vy = (det["y"] - old_y) / dt

            # Smooth velocity. This prevents jitter from raw detection noise.
            alpha = 0.65
            self.vx = alpha * self.vx + (1.0 - alpha) * meas_vx
            self.vy = alpha * self.vy + (1.0 - alpha) * meas_vy

        self.x = det["x"]
        self.y = det["y"]
        self.z = det["z"]

        self.l = det["l"]
        self.w = det["w"]
        self.h = det["h"]
        self.yaw = det["yaw"]
        self.score = det["score"]

        self.hits += 1
        self.time_since_update = 0
        self.last_update_frame = frame_idx
        self.last_match_distance = match_distance
        self.last_detection_score = det["score"]

        self.history.append((self.x, self.y))


class OnlineBEVTracker:
    def __init__(self, max_age=5, min_hits=2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.tracks = []

    def update(self, detections, frame_idx, dt):
        for trk in self.tracks:
            trk.predict(dt)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        matches = []

        candidate_pairs = []

        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if not classes_compatible(trk.class_name, det["class_name"]):
                    continue

                dx = trk.x - det["x"]
                dy = trk.y - det["y"]
                dist = math.sqrt(dx * dx + dy * dy)

                max_dist = detection_match_threshold(det["class_name"])
                if dist <= max_dist:
                    candidate_pairs.append((dist, ti, di))

        candidate_pairs.sort(key=lambda x: x[0])

        for dist, ti, di in candidate_pairs:
            if ti not in unmatched_tracks or di not in unmatched_dets:
                continue

            matches.append((ti, di, dist))
            unmatched_tracks.remove(ti)
            unmatched_dets.remove(di)

        for ti, di, dist in matches:
            self.tracks[ti].update(detections[di], frame_idx, dt, dist)

        for di in unmatched_dets:
            trk = Track(self.next_track_id, detections[di], frame_idx)
            self.next_track_id += 1
            self.tracks.append(trk)

        self.tracks = [
            trk for trk in self.tracks
            if trk.time_since_update <= self.max_age
        ]

        return self.tracks

    def active_tracks(self):
        return [
            trk for trk in self.tracks
            if trk.hits >= self.min_hits and trk.time_since_update == 0
        ]



def angle_wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a, b):
    return angle_wrap(a - b)


def angle_lerp(old, new, alpha):
    return angle_wrap(old + alpha * angle_diff(new, old))


def yaw_consistent_with_velocity(yaw, vx, vy, speed_thr=1.5):
    """
    Keep detector yaw, but resolve 180-degree front/back ambiguity using motion.
    This is much less aggressive than directly setting yaw = atan2(vy, vx).
    """
    speed = math.hypot(vx, vy)
    if speed < speed_thr:
        return yaw

    vel_yaw = math.atan2(vy, vx)

    # If detector yaw points mostly opposite to motion, flip by pi.
    if abs(angle_diff(yaw, vel_yaw)) > math.pi / 2.0:
        yaw = angle_wrap(yaw + math.pi)

    return yaw


def bev_corners_from_box(x, y, l, w, yaw):
    local = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
    ], dtype=np.float32)

    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    return local @ R.T + np.array([x, y], dtype=np.float32)


def world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H):
    px = int((x - x_min) / (x_max - x_min) * (W - 1))
    py = int((y_max - y) / (y_max - y_min) * (H - 1))
    return px, py


def track_color(track_id):
    # Stable pseudo-random color by ID, no external state needed.
    rng = np.random.default_rng(track_id * 12345)
    c = rng.integers(80, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_bev_frame(
    points,
    detections,
    tracks,
    frame_idx,
    frame_id,
    timestamp,
    x_min,
    x_max,
    y_min,
    y_max,
    W,
    H,
    trail_len,
    min_hits,
    draw_lost_tracks,
    lost_draw_age,
):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)

    # Grid
    for gx in range(int(math.floor(x_min / 20) * 20), int(x_max) + 1, 20):
        p1 = world_to_pixel(gx, y_min, x_min, x_max, y_min, y_max, W, H)
        p2 = world_to_pixel(gx, y_max, x_min, x_max, y_min, y_max, W, H)
        cv2.line(img, p1, p2, (45, 45, 45), 1)

    for gy in range(int(math.floor(y_min / 20) * 20), int(y_max) + 1, 20):
        p1 = world_to_pixel(x_min, gy, x_min, x_max, y_min, y_max, W, H)
        p2 = world_to_pixel(x_max, gy, x_min, x_max, y_min, y_max, W, H)
        cv2.line(img, p1, p2, (45, 45, 45), 1)

    # Point cloud background
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    pts = points[mask]

    if len(pts) > 0:
        # Downsample for speed
        if len(pts) > 90000:
            pts = pts[::2]

        z = pts[:, 2]
        z_norm = np.clip((z - (-8.0)) / 12.0, 0, 1)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]

        px = ((pts[:, 0] - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
        py = ((y_max - pts[:, 1]) / (y_max - y_min) * (H - 1)).astype(np.int32)

        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        img[py[valid], px[valid]] = colors[valid]

    # Raw detections as thin red boxes
    for det in detections:
        corners = bev_corners_from_box(det["x"], det["y"], det["l"], det["w"], det["yaw"])
        pts_px = np.array([
            world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H)
            for x, y in corners
        ], dtype=np.int32)

        cv2.polylines(img, [pts_px], True, (60, 60, 255), 1)

    # Tracks
    active_count = 0

    for trk in tracks:
        confirmed = trk.hits >= min_hits
        if not confirmed:
            continue

        # For the main demo, do not draw old predicted-only tracks.
        # They create clutter as gray boxes. Enable them only for debugging.
        if trk.time_since_update > 0:
            if not draw_lost_tracks:
                continue
            if trk.time_since_update > lost_draw_age:
                continue

        active_count += int(trk.time_since_update == 0)

        color = track_color(trk.track_id)
        if trk.time_since_update > 0:
            color = (120, 120, 120)

        corners = bev_corners_from_box(trk.x, trk.y, trk.l, trk.w, trk.yaw)
        pts_px = np.array([
            world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H)
            for x, y in corners
        ], dtype=np.int32)

        cv2.polylines(img, [pts_px], True, color, 3)

        cx, cy = world_to_pixel(trk.x, trk.y, x_min, x_max, y_min, y_max, W, H)

        # Velocity arrow
        speed = math.sqrt(trk.vx * trk.vx + trk.vy * trk.vy)
        vx_vis = trk.vx * 1.2
        vy_vis = trk.vy * 1.2
        ex, ey = world_to_pixel(trk.x + vx_vis, trk.y + vy_vis, x_min, x_max, y_min, y_max, W, H)

        # Trail
        hist = trk.history[-trail_len:]
        if len(hist) >= 2:
            hist_px = np.array([
                world_to_pixel(x, y, x_min, x_max, y_min, y_max, W, H)
                for x, y in hist
            ], dtype=np.int32)
            cv2.polylines(img, [hist_px], False, color, 2)

        label = f"T{trk.track_id} {trk.class_name} {speed:.1f}m/s"
        cv2.putText(
            img,
            label,
            (cx + 5, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Header
    cv2.rectangle(img, (0, 0), (W, 84), (0, 0, 0), -1)

    cv2.putText(
        img,
        f"BEV Online Tracking Replay | frame {frame_idx:03d}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"{frame_id} | t={timestamp:.3f} | detections={len(detections)} | active_tracks={active_count}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        "thin red = raw detections | colored boxes = tracks | dots = future motion | tail = trajectory",
        (W - 760, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return img


def build_frame_list(dataset_root: Path, pred_root: Path, split: str):
    gt_dir = dataset_root / split / "labels_point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
    pcd_dir = dataset_root / split / "point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"

    gt_files = sorted(gt_dir.glob("*.json"))

    frames = []

    for gt_path in gt_files:
        frame_id = gt_path.stem
        timestamp = parse_timestamp_from_frame_id(frame_id)
        pcd_path = pcd_dir / f"{frame_id}.pcd"
        pred_path = pred_root / f"{frame_id}.json"

        frames.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "gt_path": gt_path,
            "pcd_path": pcd_path,
            "pred_path": pred_path,
            "has_pred": pred_path.exists(),
        })

    frames.sort(key=lambda x: x["timestamp"])
    return frames


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", default="external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset")
    parser.add_argument("--pred-root", default="external/coopdet3d/inference_val_openlabel/openlabel")
    parser.add_argument("--out-dir", default="outputs/tracking_replay/pretrained_val")
    parser.add_argument("--split", default="val")

    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--trail-len", type=int, default=20)
    parser.add_argument("--reset-gap-sec", type=float, default=1.5)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=-1)

    parser.add_argument("--x-min", type=float, default=-80)
    parser.add_argument("--x-max", type=float, default=120)
    parser.add_argument("--y-min", type=float, default=-100)
    parser.add_argument("--y-max", type=float, default=120)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=float, default=10.0)

    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--draw-lost-tracks", action="store_true")
    parser.add_argument("--lost-draw-age", type=int, default=2)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    frames = build_frame_list(dataset_root, pred_root, args.split)

    if not frames:
        raise RuntimeError("No frames found. Check dataset root and split.")

    frames = frames[args.start_index:]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    print(f"Loaded frames: {len(frames)}")
    print(f"Prediction root: {pred_root}")

    tracker = OnlineBEVTracker(max_age=args.max_age, min_hits=args.min_hits)

    video_writer = None
    video_path = out_dir / "tracking.mp4"

    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (args.width, args.height))

    csv_path = out_dir / "diagnostics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "frame_idx",
        "frame_id",
        "timestamp",
        "dt",
        "raw_dt",
        "sequence_reset",
        "track_id",
        "class_name",
        "x",
        "y",
        "z",
        "l",
        "w",
        "h",
        "yaw",
        "vx",
        "vy",
        "speed_mps",
        "age",
        "hits",
        "time_since_update",
        "last_detection_score",
        "last_match_distance",
        "num_detections",
        "has_pred_file",
    ])
    csv_writer.writeheader()

    prev_timestamp = None

    for local_idx, frame in enumerate(frames):
        frame_idx = args.start_index + local_idx
        frame_id = frame["frame_id"]
        timestamp = frame["timestamp"]

        sequence_reset = False

        if prev_timestamp is None:
            raw_dt = 0.1
            dt = 0.1
        else:
            raw_dt = timestamp - prev_timestamp
            dt = max(1e-3, raw_dt)

            # Validation frames are not guaranteed to be one continuous video.
            # If the gap is large, reset the tracker so IDs do not carry into a new clip.
            if raw_dt > args.reset_gap_sec:
                print(f"RESET tracker at frame {frame_idx}: timestamp gap {raw_dt:.3f}s > reset_gap_sec={args.reset_gap_sec:.3f}s")
                tracker = OnlineBEVTracker(max_age=args.max_age, min_hits=args.min_hits)
                sequence_reset = True
                dt = 0.1

        prev_timestamp = timestamp

        if not frame["pcd_path"].exists():
            print(f"SKIP missing PCD: {frame['pcd_path']}")
            continue

        points = load_pcd_ascii(frame["pcd_path"])
        detections = parse_openlabel_boxes(frame["pred_path"], score_thr=args.score_thr)

        tracks = tracker.update(detections, frame_idx=frame_idx, dt=dt)

        img = draw_bev_frame(
            points=points,
            detections=detections,
            tracks=tracks,
            frame_idx=frame_idx,
            frame_id=frame_id,
            timestamp=timestamp,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            W=args.width,
            H=args.height,
            trail_len=args.trail_len,
            min_hits=args.min_hits,
            draw_lost_tracks=args.draw_lost_tracks,
            lost_draw_age=args.lost_draw_age,
        )

        if video_writer is not None:
            video_writer.write(img)

        if args.save_frames:
            out_frame = frames_dir / f"{frame_idx:04d}_{frame_id}.jpg"
            cv2.imwrite(str(out_frame), img)

        for trk in tracks:
            if trk.hits < args.min_hits:
                continue

            # For first diagnostics, only log tracks that were updated by a detection
            # in the current frame. Lost/predicted-only tracks are shown in gray in the video,
            # but they are not counted as active tracking output in the CSV.
            if trk.time_since_update != 0:
                continue

            speed = math.sqrt(trk.vx * trk.vx + trk.vy * trk.vy)

            csv_writer.writerow({
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "dt": dt,
                "raw_dt": raw_dt,
                "sequence_reset": sequence_reset,
                "track_id": trk.track_id,
                "class_name": trk.class_name,
                "x": trk.x,
                "y": trk.y,
                "z": trk.z,
                "l": trk.l,
                        "w": trk.w,
                        "h": trk.h,
                        "yaw": trk.yaw,
                        "vx": trk.vx,
                "vy": trk.vy,
                "speed_mps": speed,
                "age": trk.age,
                "hits": trk.hits,
                "time_since_update": trk.time_since_update,
                "last_detection_score": trk.last_detection_score,
                "last_match_distance": trk.last_match_distance,
                "num_detections": len(detections),
                "has_pred_file": frame["has_pred"],
            })

        print(
            f"{frame_idx:03d} | det={len(detections):02d} | tracks={len(tracks):02d} | "
            f"active={len(tracker.active_tracks()):02d} | pred_file={frame['has_pred']}"
        )

    csv_file.close()

    if video_writer is not None:
        video_writer.release()

    print("\nDone.")
    print(f"Output dir: {out_dir}")
    print(f"Video: {video_path if not args.no_video else 'disabled'}")
    print(f"Diagnostics CSV: {csv_path}")


if __name__ == "__main__":
    main()
