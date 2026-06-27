#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


VEHICLE_KEYWORDS = {
    "car", "truck", "bus", "van", "trailer", "vehicle",
    "emergency_vehicle", "emergency-vehicle",
    "motor_vehicle", "motor-vehicle",
}

VRU_KEYWORDS = {
    "pedestrian", "person", "bicycle", "cyclist",
    "bike", "motorcycle", "motorbike",
}

SEVERITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def norm_class(name):
    if pd.isna(name):
        return "unknown"
    return str(name).strip().lower().replace(" ", "_")


def is_vehicle(cls):
    c = norm_class(cls)
    return any(k in c for k in VEHICLE_KEYWORDS)


def is_vru(cls):
    c = norm_class(cls)
    return any(k in c for k in VRU_KEYWORDS)


def to_builtin(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def jsonable_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = [to_builtin(x) for x in v]
        else:
            out[k] = to_builtin(v)
    return out


def connected_components(points_xy, threshold):
    """
    Simple distance-threshold clustering.
    points_xy: Nx2 array
    returns list of index lists
    """
    n = len(points_xy)
    if n == 0:
        return []

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points_xy[i] - points_xy[j]))
            if d <= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return list(groups.values())


def pairwise_path_min(vehicle_path, group_path, times):
    """
    Computes:
    1. same-time minimum distance
    2. any-path minimum distance, allowing slight time mismatch
    """
    same = np.linalg.norm(vehicle_path - group_path, axis=1)
    same_idx = int(np.argmin(same))

    # Any point on vehicle sampled path vs any point on group sampled path.
    diff = vehicle_path[:, None, :] - group_path[None, :, :]
    dmat = np.linalg.norm(diff, axis=2)
    flat_idx = int(np.argmin(dmat))
    vi, gi = np.unravel_index(flat_idx, dmat.shape)

    return {
        "same_time_min_distance": float(same[same_idx]),
        "same_time_of_min": float(times[same_idx]),
        "path_min_distance": float(dmat[vi, gi]),
        "vehicle_time_of_path_min": float(times[vi]),
        "group_time_of_path_min": float(times[gi]),
        "time_gap_at_path_min": float(abs(times[vi] - times[gi])),
    }


def severity_from_distance(d, high_thr, med_thr):
    if d < high_thr:
        return "HIGH"
    if d < med_thr:
        return "MEDIUM"
    return "LOW"


def make_frame_events(df, args):
    events = []

    times = np.arange(0.0, args.prediction_horizon + 1e-9, args.prediction_step)

    for (frame_idx, frame_id, timestamp), g in df.groupby(["frame_idx", "frame_id", "timestamp"], sort=True):
        frame_rows = list(g.to_dict("records"))

        vrus = [r for r in frame_rows if is_vru(r["class_name"])]
        vehicles = [r for r in frame_rows if is_vehicle(r["class_name"])]

        if len(vrus) < args.min_group_size or len(vehicles) == 0:
            continue

        vru_xy = np.array([[float(r["x"]), float(r["y"])] for r in vrus], dtype=np.float64)
        clusters = connected_components(vru_xy, args.cluster_distance)

        for cluster_i, idxs in enumerate(clusters):
            if len(idxs) < args.min_group_size:
                continue

            group_rows = [vrus[i] for i in idxs]
            group_track_ids = sorted([int(r["track_id"]) for r in group_rows])

            gx = float(np.mean([float(r["x"]) for r in group_rows]))
            gy = float(np.mean([float(r["y"]) for r in group_rows]))
            gvx = float(np.mean([float(r["vx"]) for r in group_rows]))
            gvy = float(np.mean([float(r["vy"]) for r in group_rows]))

            group_pos = np.array([gx, gy], dtype=np.float64)
            group_vel = np.array([gvx, gvy], dtype=np.float64)
            group_speed = float(np.linalg.norm(group_vel))

            if group_speed < args.min_group_speed:
                continue

            group_path = group_pos[None, :] + times[:, None] * group_vel[None, :]

            for veh in vehicles:
                veh_tid = int(veh["track_id"])
                veh_cls = norm_class(veh["class_name"])

                vx = float(veh["x"])
                vy = float(veh["y"])
                vvx = float(veh["vx"])
                vvy = float(veh["vy"])

                veh_pos = np.array([vx, vy], dtype=np.float64)
                veh_vel = np.array([vvx, vvy], dtype=np.float64)
                veh_speed = float(np.linalg.norm(veh_vel))

                if veh_speed < args.min_vehicle_speed:
                    continue

                current_dist = float(np.linalg.norm(veh_pos - group_pos))
                if current_dist > args.max_current_distance:
                    continue

                veh_path = veh_pos[None, :] + times[:, None] * veh_vel[None, :]
                path_stats = pairwise_path_min(veh_path, group_path, times)

                same_min = path_stats["same_time_min_distance"]
                path_min = path_stats["path_min_distance"]
                time_gap = path_stats["time_gap_at_path_min"]

                same_time_conflict = same_min <= args.same_time_conflict_distance
                path_conflict = (
                    path_min <= args.path_conflict_distance
                    and time_gap <= args.max_path_time_gap
                )

                if not (same_time_conflict or path_conflict):
                    continue

                # Vehicle should be moving roughly toward the group/future group area.
                # Negative dot product means relative distance is decreasing.
                rel_pos = group_pos - veh_pos
                rel_vel = group_vel - veh_vel
                approaching_score = float(np.dot(rel_pos, rel_vel))
                approaching = approaching_score < -args.approach_margin

                getting_closer = current_dist - same_min >= args.getting_closer_margin

                if not (approaching or getting_closer or current_dist <= args.same_time_conflict_distance):
                    continue

                risk_dist = min(same_min, path_min)
                severity = severity_from_distance(
                    risk_dist,
                    high_thr=args.high_distance,
                    med_thr=args.medium_distance,
                )

                event_id = f"{int(frame_idx)}_CROSSING_GROUP_VEHICLE_RISK_V{veh_tid}_G{'-'.join(map(str, group_track_ids))}"

                desc = (
                    f"Crossing group risk: vehicle T{veh_tid} ({veh_cls}) near VRU group "
                    f"{group_track_ids}; group_size={len(group_track_ids)}, "
                    f"current={current_dist:.1f}m, same_min={same_min:.1f}m, path_min={path_min:.1f}m"
                )

                events.append({
                    "event_id": event_id,
                    "frame_idx": int(frame_idx),
                    "frame_id": str(frame_id),
                    "timestamp": float(timestamp),
                    "event_type": "CROSSING_GROUP_VEHICLE_RISK",
                    "severity": severity,
                    "vehicle_track_id": veh_tid,
                    "vehicle_class": veh_cls,
                    "group_track_ids": group_track_ids,
                    "group_size": int(len(group_track_ids)),
                    "group_center_x": round(gx, 3),
                    "group_center_y": round(gy, 3),
                    "group_vx": round(gvx, 3),
                    "group_vy": round(gvy, 3),
                    "group_speed_mps": round(group_speed, 3),
                    "vehicle_x": round(vx, 3),
                    "vehicle_y": round(vy, 3),
                    "vehicle_vx": round(vvx, 3),
                    "vehicle_vy": round(vvy, 3),
                    "vehicle_speed_mps": round(veh_speed, 3),
                    "current_distance_m": round(current_dist, 3),
                    "same_time_min_distance_m": round(same_min, 3),
                    "same_time_of_min_s": round(path_stats["same_time_of_min"], 3),
                    "path_min_distance_m": round(path_min, 3),
                    "vehicle_time_of_path_min_s": round(path_stats["vehicle_time_of_path_min"], 3),
                    "group_time_of_path_min_s": round(path_stats["group_time_of_path_min"], 3),
                    "time_gap_at_path_min_s": round(time_gap, 3),
                    "risk_distance_m": round(risk_dist, 3),
                    "approaching": bool(approaching),
                    "description": desc,
                })

    events = [jsonable_dict(e) for e in events]

    events = sorted(
        events,
        key=lambda e: (
            int(e["frame_idx"]),
            SEVERITY_RANK.get(e["severity"], 99),
            int(e["vehicle_track_id"]),
            tuple(e["group_track_ids"]),
        ),
    )

    return events


def group_overlap(a, b):
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_episodes(events, min_event_frames=3, merge_gap_frames=5, min_group_overlap=0.25):
    if not events:
        return []

    # Bucket by vehicle id first. Group ids can slightly change frame to frame.
    buckets = {}
    for e in events:
        buckets.setdefault(int(e["vehicle_track_id"]), []).append(e)

    episodes = []
    ep_counter = 0

    for veh_tid, evs in buckets.items():
        evs = sorted(evs, key=lambda e: (int(e["frame_idx"]), tuple(e["group_track_ids"])))

        current = []
        last_frame = None
        last_group = None

        def flush(chunk):
            nonlocal ep_counter

            if not chunk:
                return

            unique_frames = sorted({int(e["frame_idx"]) for e in chunk})
            if len(unique_frames) < min_event_frames:
                return

            chunk = sorted(chunk, key=lambda e: (int(e["frame_idx"]), float(e["risk_distance_m"])))

            start = chunk[0]
            end = chunk[-1]
            risk_event = min(chunk, key=lambda e: (float(e["risk_distance_m"]), int(e["frame_idx"])))

            all_group_ids = sorted(set(itertools.chain.from_iterable(e["group_track_ids"] for e in chunk)))
            best_severity = sorted(
                {e["severity"] for e in chunk},
                key=lambda s: SEVERITY_RANK.get(s, 99),
            )[0]

            ep_counter += 1
            ep_id = f"GE{ep_counter:04d}_CROSSING_GROUP_RISK_V{veh_tid}_G{'-'.join(map(str, all_group_ids))}"

            duration_s = max(0.0, float(end["timestamp"]) - float(start["timestamp"]))

            desc = (
                f"Crossing group risk episode: vehicle T{veh_tid} with VRU group {all_group_ids}; "
                f"frames {start['frame_idx']}-{end['frame_idx']}, "
                f"severity={best_severity}, risk_dist={float(risk_event['risk_distance_m']):.1f}m"
            )

            episodes.append({
                "episode_id": ep_id,
                "event_type": "CROSSING_GROUP_VEHICLE_RISK",
                "severity": best_severity,
                "vehicle_track_id": int(veh_tid),
                "group_track_ids": all_group_ids,
                "group_size_max": int(max(e["group_size"] for e in chunk)),
                "start_frame_idx": int(start["frame_idx"]),
                "end_frame_idx": int(end["frame_idx"]),
                "start_frame_id": str(start["frame_id"]),
                "end_frame_id": str(end["frame_id"]),
                "start_timestamp": round(float(start["timestamp"]), 6),
                "end_timestamp": round(float(end["timestamp"]), 6),
                "duration_s": round(duration_s, 3),
                "num_event_frames": int(len(unique_frames)),
                "frame_of_min_risk": int(risk_event["frame_idx"]),
                "risk_distance_m": round(float(risk_event["risk_distance_m"]), 3),
                "same_time_min_distance_m": round(float(risk_event["same_time_min_distance_m"]), 3),
                "path_min_distance_m": round(float(risk_event["path_min_distance_m"]), 3),
                "current_distance_at_min_risk_m": round(float(risk_event["current_distance_m"]), 3),
                "vehicle_speed_at_min_risk_mps": round(float(risk_event["vehicle_speed_mps"]), 3),
                "group_speed_at_min_risk_mps": round(float(risk_event["group_speed_mps"]), 3),
                "group_center_x_at_min_risk": round(float(risk_event["group_center_x"]), 3),
                "group_center_y_at_min_risk": round(float(risk_event["group_center_y"]), 3),
                "group_vx_at_min_risk": round(float(risk_event["group_vx"]), 3),
                "group_vy_at_min_risk": round(float(risk_event["group_vy"]), 3),
                "vehicle_x_at_min_risk": round(float(risk_event["vehicle_x"]), 3),
                "vehicle_y_at_min_risk": round(float(risk_event["vehicle_y"]), 3),
                "vehicle_vx_at_min_risk": round(float(risk_event["vehicle_vx"]), 3),
                "vehicle_vy_at_min_risk": round(float(risk_event["vehicle_vy"]), 3),
                "description": desc,
            })

        for e in evs:
            f = int(e["frame_idx"])
            g = e["group_track_ids"]

            if last_frame is None:
                current = [e]
                last_frame = f
                last_group = g
                continue

            frame_ok = f - last_frame <= merge_gap_frames
            group_ok = group_overlap(g, last_group) >= min_group_overlap

            if frame_ok and group_ok:
                current.append(e)
            else:
                flush(current)
                current = [e]

            last_frame = f
            last_group = g

        flush(current)

    episodes = [jsonable_dict(e) for e in episodes]

    episodes = sorted(
        episodes,
        key=lambda e: (
            SEVERITY_RANK.get(e["severity"], 99),
            int(e["start_frame_idx"]),
            float(e["risk_distance_m"]),
        ),
    )

    return episodes


def save_json_payload(events, episodes, out_dir, tracking_csv, args):
    frame_events = {}
    for e in events:
        frame_events.setdefault(str(e["frame_idx"]), []).append(e)

    active = {}
    for ep in episodes:
        for f in range(int(ep["start_frame_idx"]), int(ep["end_frame_idx"]) + 1):
            active.setdefault(str(f), []).append(ep["episode_id"])

    payload = {
        "source_tracking_csv": str(tracking_csv),
        "params": vars(args),
        "num_frame_events": len(events),
        "num_episodes": len(episodes),
        "frame_events_by_frame_idx": frame_events,
        "episodes": episodes,
        "active_episode_ids_by_frame_idx": active,
    }

    with open(out_dir / "group_crossing_events.json", "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tracking-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--cluster-distance", type=float, default=5.0)
    ap.add_argument("--min-group-size", type=int, default=2)

    ap.add_argument("--prediction-horizon", type=float, default=5.0)
    ap.add_argument("--prediction-step", type=float, default=0.5)

    ap.add_argument("--min-group-speed", type=float, default=0.15)
    ap.add_argument("--min-vehicle-speed", type=float, default=0.8)
    ap.add_argument("--max-current-distance", type=float, default=22.0)

    ap.add_argument("--same-time-conflict-distance", type=float, default=5.0)
    ap.add_argument("--path-conflict-distance", type=float, default=4.0)
    ap.add_argument("--max-path-time-gap", type=float, default=2.0)

    ap.add_argument("--high-distance", type=float, default=2.0)
    ap.add_argument("--medium-distance", type=float, default=4.5)

    ap.add_argument("--approach-margin", type=float, default=0.3)
    ap.add_argument("--getting-closer-margin", type=float, default=0.8)

    ap.add_argument("--max-time-since-update", type=int, default=1)

    ap.add_argument("--min-event-frames", type=int, default=3)
    ap.add_argument("--merge-gap-frames", type=int, default=5)
    ap.add_argument("--min-group-overlap", type=float, default=0.25)

    args = ap.parse_args()

    tracking_csv = Path(args.tracking_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracking_csv)

    required = ["frame_idx", "frame_id", "timestamp", "track_id", "class_name", "x", "y", "vx", "vy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    if "speed_mps" not in df.columns:
        df["speed_mps"] = np.sqrt(df["vx"].astype(float) ** 2 + df["vy"].astype(float) ** 2)

    if "time_since_update" in df.columns:
        df = df[df["time_since_update"].fillna(999).astype(float) <= args.max_time_since_update].copy()

    for c in ["x", "y", "vx", "vy", "speed_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["frame_idx", "track_id", "x", "y", "vx", "vy", "speed_mps"]).copy()

    events = make_frame_events(df, args)
    episodes = build_episodes(
        events,
        min_event_frames=args.min_event_frames,
        merge_gap_frames=args.merge_gap_frames,
        min_group_overlap=args.min_group_overlap,
    )

    events_csv = out_dir / "group_crossing_frame_events.csv"
    episodes_csv = out_dir / "group_crossing_episodes.csv"

    pd.DataFrame(events).to_csv(events_csv, index=False)
    pd.DataFrame(episodes).to_csv(episodes_csv, index=False)
    save_json_payload(events, episodes, out_dir, tracking_csv, args)

    print("Saved:")
    print(f"  {events_csv}")
    print(f"  {episodes_csv}")
    print(f"  {out_dir / 'group_crossing_events.json'}")
    print()

    print(f"Frame-level group crossing events: {len(events)}")
    print(f"Group crossing episodes: {len(episodes)}")
    print()

    if episodes:
        edf = pd.DataFrame(episodes)
        print(edf[[
            "episode_id",
            "severity",
            "vehicle_track_id",
            "group_track_ids",
            "start_frame_idx",
            "end_frame_idx",
            "num_event_frames",
            "risk_distance_m",
            "frame_of_min_risk",
            "description",
        ]].to_string(index=False))
    else:
        print("No group crossing episodes found.")
        print("Try looser thresholds:")
        print("  --same-time-conflict-distance 6.0 --path-conflict-distance 5.0 --max-current-distance 28.0")


if __name__ == "__main__":
    main()
