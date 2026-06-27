#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "frame_idx",
    "frame_id",
    "timestamp",
    "event_type",
    "severity",
    "track_id_1",
    "track_id_2",
    "class_1",
    "class_2",
    "distance_m",
    "min_pred_distance_m",
    "time_of_min_distance_s",
    "speed_1_mps",
    "speed_2_mps",
    "approaching",
    "description",
]


EPISODE_COLUMNS = [
    "episode_id",
    "event_type",
    "severity",
    "track_id_1",
    "track_id_2",
    "class_1",
    "class_2",
    "start_frame_idx",
    "end_frame_idx",
    "start_frame_id",
    "end_frame_id",
    "start_timestamp",
    "end_timestamp",
    "duration_s",
    "num_event_frames",
    "min_distance_m",
    "min_pred_distance_m",
    "time_of_min_distance_s",
    "frame_of_min_risk",
    "max_speed_1_mps",
    "max_speed_2_mps",
    "description",
]


VEHICLE_KEYWORDS = {
    "car",
    "truck",
    "bus",
    "van",
    "trailer",
    "vehicle",
    "emergency_vehicle",
    "emergency-vehicle",
    "motor_vehicle",
    "motor-vehicle",
}

VRU_KEYWORDS = {
    "pedestrian",
    "person",
    "bicycle",
    "cyclist",
    "bike",
    "motorcycle",
    "motorbike",
}


SEVERITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
EVENT_RANK = {"POTENTIAL_CONFLICT": 0, "PEDESTRIAN_NEAR_VEHICLE": 1}


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


def severity_for_distance(distance_m, high_thr, medium_thr):
    if distance_m < high_thr:
        return "HIGH"
    if distance_m < medium_thr:
        return "MEDIUM"
    return "LOW"


def to_builtin(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def jsonable_dict(d):
    return {k: to_builtin(v) for k, v in d.items()}


def make_event(
    frame_idx,
    frame_id,
    timestamp,
    event_type,
    severity,
    t1,
    t2,
    cls1,
    cls2,
    distance_m,
    min_pred_distance_m,
    time_of_min_distance_s,
    speed1,
    speed2,
    approaching,
    description,
):
    event_id = f"{int(frame_idx)}_{event_type}_T{int(t1)}_T{int(t2)}"

    return {
        "event_id": event_id,
        "frame_idx": int(frame_idx),
        "frame_id": str(frame_id),
        "timestamp": float(timestamp),
        "event_type": event_type,
        "severity": severity,
        "track_id_1": int(t1),
        "track_id_2": int(t2),
        "class_1": str(cls1),
        "class_2": str(cls2),
        "distance_m": round(float(distance_m), 3),
        "min_pred_distance_m": round(float(min_pred_distance_m), 3),
        "time_of_min_distance_s": round(float(time_of_min_distance_s), 3),
        "speed_1_mps": round(float(speed1), 3),
        "speed_2_mps": round(float(speed2), 3),
        "approaching": bool(approaching),
        "description": description,
    }


def generate_frame_events(df, args):
    t_samples = np.arange(0.0, args.prediction_horizon + 1e-9, args.prediction_step)
    events = []

    group_cols = ["frame_idx", "frame_id", "timestamp"]

    for (frame_idx, frame_id, timestamp), g in df.groupby(group_cols, sort=True):
        rows = list(g.to_dict("records"))

        for a, b in itertools.combinations(rows, 2):
            tid1 = int(a["track_id"])
            tid2 = int(b["track_id"])

            cls1 = norm_class(a["class_name"])
            cls2 = norm_class(b["class_name"])

            a_vehicle = is_vehicle(cls1)
            b_vehicle = is_vehicle(cls2)
            a_vru = is_vru(cls1)
            b_vru = is_vru(cls2)

            has_vehicle = a_vehicle or b_vehicle
            if not has_vehicle:
                continue

            p1 = np.array([float(a["x"]), float(a["y"])], dtype=np.float64)
            p2 = np.array([float(b["x"]), float(b["y"])], dtype=np.float64)

            v1 = np.array([float(a["vx"]), float(a["vy"])], dtype=np.float64)
            v2 = np.array([float(b["vx"]), float(b["vy"])], dtype=np.float64)

            speed1 = float(a["speed_mps"])
            speed2 = float(b["speed_mps"])

            current_distance = float(np.linalg.norm(p2 - p1))

            future_distances = []
            for t in t_samples:
                fp1 = p1 + v1 * t
                fp2 = p2 + v2 * t
                future_distances.append(float(np.linalg.norm(fp2 - fp1)))

            future_distances = np.asarray(future_distances)
            min_idx = int(np.argmin(future_distances))
            min_pred_distance = float(future_distances[min_idx])
            time_of_min = float(t_samples[min_idx])

            rel_pos = p2 - p1
            rel_vel = v2 - v1
            approaching_score = float(np.dot(rel_pos, rel_vel))
            approaching = approaching_score < -args.approach_margin
            getting_closer = current_distance - min_pred_distance >= args.getting_closer_margin

            pair_events = []

            # ---------------------------------------------------------
            # 1. POTENTIAL_CONFLICT
            # ---------------------------------------------------------
            allowed_conflict_pair = (
                (a_vehicle and b_vehicle)
                or (a_vehicle and b_vru)
                or (b_vehicle and a_vru)
            )

            moving_enough = max(speed1, speed2) >= args.min_moving_speed

            conflict_condition = (
                allowed_conflict_pair
                and moving_enough
                and current_distance <= args.conflict_current_distance_max
                and min_pred_distance < args.conflict_distance
                and (approaching or getting_closer or current_distance < args.conflict_distance)
            )

            if conflict_condition:
                severity = severity_for_distance(
                    min_pred_distance,
                    high_thr=args.conflict_high_distance,
                    medium_thr=args.conflict_distance,
                )

                desc = (
                    f"Potential conflict between T{tid1} ({cls1}) and T{tid2} ({cls2}); "
                    f"current={current_distance:.1f}m, "
                    f"min_pred={min_pred_distance:.1f}m at {time_of_min:.1f}s"
                )

                pair_events.append(make_event(
                    frame_idx=frame_idx,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    event_type="POTENTIAL_CONFLICT",
                    severity=severity,
                    t1=tid1,
                    t2=tid2,
                    cls1=cls1,
                    cls2=cls2,
                    distance_m=current_distance,
                    min_pred_distance_m=min_pred_distance,
                    time_of_min_distance_s=time_of_min,
                    speed1=speed1,
                    speed2=speed2,
                    approaching=approaching,
                    description=desc,
                ))

            # ---------------------------------------------------------
            # 2. PEDESTRIAN_NEAR_VEHICLE
            # ---------------------------------------------------------
            ped_near_condition = (
                ((a_vehicle and b_vru) or (b_vehicle and a_vru))
                and current_distance < args.ped_vehicle_distance
            )

            if ped_near_condition:
                # Put VRU first and vehicle second for readability.
                if a_vru and b_vehicle:
                    ped_tid, veh_tid = tid1, tid2
                    ped_cls, veh_cls = cls1, cls2
                    ped_speed, veh_speed = speed1, speed2
                else:
                    ped_tid, veh_tid = tid2, tid1
                    ped_cls, veh_cls = cls2, cls1
                    ped_speed, veh_speed = speed2, speed1

                severity = severity_for_distance(
                    current_distance,
                    high_thr=args.ped_high_distance,
                    medium_thr=args.ped_medium_distance,
                )

                desc = (
                    f"VRU T{ped_tid} ({ped_cls}) near vehicle T{veh_tid} ({veh_cls}); "
                    f"distance={current_distance:.1f}m"
                )

                near_event = make_event(
                    frame_idx=frame_idx,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    event_type="PEDESTRIAN_NEAR_VEHICLE",
                    severity=severity,
                    t1=ped_tid,
                    t2=veh_tid,
                    cls1=ped_cls,
                    cls2=veh_cls,
                    distance_m=current_distance,
                    min_pred_distance_m=min_pred_distance,
                    time_of_min_distance_s=time_of_min,
                    speed1=ped_speed,
                    speed2=veh_speed,
                    approaching=approaching,
                    description=desc,
                )

                # If conflict exists for the same pair in the same frame,
                # suppress proximity by default.
                if args.emit_both or not any(e["event_type"] == "POTENTIAL_CONFLICT" for e in pair_events):
                    pair_events.append(near_event)

            events.extend(pair_events)

    events = [jsonable_dict(e) for e in events]

    events = sorted(
        events,
        key=lambda e: (
            e["frame_idx"],
            EVENT_RANK.get(e["event_type"], 99),
            SEVERITY_RANK.get(e["severity"], 99),
            e["track_id_1"],
            e["track_id_2"],
        ),
    )

    return events


def episode_key(event):
    # Merge same type + same pair. Pair order should not split potential conflicts.
    tid_a = int(event["track_id_1"])
    tid_b = int(event["track_id_2"])
    pair = tuple(sorted([tid_a, tid_b]))
    return event["event_type"], pair[0], pair[1]


def build_episodes(events, min_event_frames=3, merge_gap_frames=5):
    if not events:
        return []

    buckets = {}
    for e in events:
        buckets.setdefault(episode_key(e), []).append(e)

    episodes = []
    episode_counter = 0

    for key, evs in buckets.items():
        evs = sorted(evs, key=lambda e: (int(e["frame_idx"]), e["timestamp"]))

        current = []
        last_frame = None

        def flush_episode(chunk):
            nonlocal episode_counter
            if not chunk:
                return

            unique_frames = sorted({int(e["frame_idx"]) for e in chunk})
            if len(unique_frames) < min_event_frames:
                return

            chunk_sorted = sorted(chunk, key=lambda e: (int(e["frame_idx"]), e["timestamp"]))

            start = chunk_sorted[0]
            end = chunk_sorted[-1]

            # Highest severity in episode.
            best_severity = sorted(
                {e["severity"] for e in chunk_sorted},
                key=lambda s: SEVERITY_RANK.get(s, 99),
            )[0]

            # Representative minimum risk event.
            risk_event = min(
                chunk_sorted,
                key=lambda e: (
                    float(e["min_pred_distance_m"]),
                    float(e["distance_m"]),
                    int(e["frame_idx"]),
                ),
            )

            episode_counter += 1
            episode_id = f"E{episode_counter:04d}_{key[0]}_T{key[1]}_T{key[2]}"

            min_distance = min(float(e["distance_m"]) for e in chunk_sorted)
            min_pred_distance = min(float(e["min_pred_distance_m"]) for e in chunk_sorted)
            max_speed_1 = max(float(e["speed_1_mps"]) for e in chunk_sorted)
            max_speed_2 = max(float(e["speed_2_mps"]) for e in chunk_sorted)

            duration_s = max(0.0, float(end["timestamp"]) - float(start["timestamp"]))

            desc = (
                f"{key[0]} episode T{start['track_id_1']} ↔ T{start['track_id_2']} "
                f"from frame {start['frame_idx']} to {end['frame_idx']}; "
                f"severity={best_severity}, min_pred={min_pred_distance:.1f}m"
            )

            episodes.append({
                "episode_id": episode_id,
                "event_type": key[0],
                "severity": best_severity,
                "track_id_1": int(start["track_id_1"]),
                "track_id_2": int(start["track_id_2"]),
                "class_1": str(start["class_1"]),
                "class_2": str(start["class_2"]),
                "start_frame_idx": int(start["frame_idx"]),
                "end_frame_idx": int(end["frame_idx"]),
                "start_frame_id": str(start["frame_id"]),
                "end_frame_id": str(end["frame_id"]),
                "start_timestamp": round(float(start["timestamp"]), 6),
                "end_timestamp": round(float(end["timestamp"]), 6),
                "duration_s": round(duration_s, 3),
                "num_event_frames": int(len(unique_frames)),
                "min_distance_m": round(float(min_distance), 3),
                "min_pred_distance_m": round(float(min_pred_distance), 3),
                "time_of_min_distance_s": round(float(risk_event["time_of_min_distance_s"]), 3),
                "frame_of_min_risk": int(risk_event["frame_idx"]),
                "max_speed_1_mps": round(float(max_speed_1), 3),
                "max_speed_2_mps": round(float(max_speed_2), 3),
                "description": desc,
            })

        for e in evs:
            f = int(e["frame_idx"])
            if last_frame is None:
                current = [e]
                last_frame = f
                continue

            if f - last_frame <= merge_gap_frames:
                current.append(e)
            else:
                flush_episode(current)
                current = [e]

            last_frame = f

        flush_episode(current)

    episodes = [jsonable_dict(e) for e in episodes]

    episodes = sorted(
        episodes,
        key=lambda e: (
            EVENT_RANK.get(e["event_type"], 99),
            SEVERITY_RANK.get(e["severity"], 99),
            int(e["start_frame_idx"]),
            float(e["min_pred_distance_m"]),
        ),
    )

    return episodes


def save_frame_events_json(events, out_path, tracking_csv, args):
    by_frame_idx = {}
    by_frame_id = {}

    for e in events:
        by_frame_idx.setdefault(str(e["frame_idx"]), []).append(e)
        by_frame_id.setdefault(str(e["frame_id"]), []).append(e)

    payload = {
        "source_tracking_csv": str(tracking_csv),
        "params": vars(args),
        "num_events": len(events),
        "events_by_frame_idx": by_frame_idx,
        "events_by_frame_id": by_frame_id,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_event_episodes_json(episodes, out_path, tracking_csv, args):
    active_by_frame_idx = {}
    active_by_frame_id = {}

    for ep in episodes:
        ep_id = ep["episode_id"]
        for f in range(int(ep["start_frame_idx"]), int(ep["end_frame_idx"]) + 1):
            active_by_frame_idx.setdefault(str(f), []).append(ep_id)

        # Frame IDs are not easy to enumerate without a full frame-id map, so use
        # start/end IDs and frame_idx map for now. Viewer can use frame_idx.
        active_by_frame_id.setdefault(str(ep["start_frame_id"]), []).append(ep_id)

    payload = {
        "source_tracking_csv": str(tracking_csv),
        "params": vars(args),
        "num_episodes": len(episodes),
        "episodes": episodes,
        "active_episode_ids_by_frame_idx": active_by_frame_idx,
        "active_episode_ids_by_frame_id": active_by_frame_id,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tracking-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--prediction-horizon", type=float, default=3.0)
    ap.add_argument("--prediction-step", type=float, default=0.5)

    # Demo-grade defaults: stricter than the initial exploratory run.
    ap.add_argument("--ped-vehicle-distance", type=float, default=4.0)
    ap.add_argument("--ped-high-distance", type=float, default=2.5)
    ap.add_argument("--ped-medium-distance", type=float, default=4.0)

    ap.add_argument("--conflict-distance", type=float, default=2.0)
    ap.add_argument("--conflict-high-distance", type=float, default=1.2)
    ap.add_argument("--conflict-current-distance-max", type=float, default=10.0)
    ap.add_argument("--min-moving-speed", type=float, default=1.2)
    ap.add_argument("--approach-margin", type=float, default=0.5)
    ap.add_argument("--getting-closer-margin", type=float, default=0.7)

    ap.add_argument("--max-time-since-update", type=int, default=1)

    # Episode aggregation.
    ap.add_argument("--min-event-frames", type=int, default=3)
    ap.add_argument("--merge-gap-frames", type=int, default=5)

    # If same pair has conflict and proximity in same frame, default keeps conflict only.
    ap.add_argument("--emit-both", action="store_true")

    args = ap.parse_args()

    tracking_csv = Path(args.tracking_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracking_csv)

    required = ["frame_idx", "frame_id", "timestamp", "track_id", "class_name", "x", "y", "vx", "vy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in diagnostics CSV: {missing}")

    if "speed_mps" not in df.columns:
        df["speed_mps"] = np.sqrt(df["vx"].astype(float) ** 2 + df["vy"].astype(float) ** 2)

    if "time_since_update" in df.columns:
        df = df[df["time_since_update"].fillna(999).astype(float) <= args.max_time_since_update].copy()

    for c in ["x", "y", "vx", "vy", "speed_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["frame_idx", "track_id", "x", "y", "vx", "vy", "speed_mps"]).copy()
    df["class_norm"] = df["class_name"].apply(norm_class)

    events = generate_frame_events(df, args)
    episodes = build_episodes(
        events,
        min_event_frames=args.min_event_frames,
        merge_gap_frames=args.merge_gap_frames,
    )

    events_csv = out_dir / "events.csv"
    frame_events_json = out_dir / "frame_events.json"
    episodes_csv = out_dir / "event_episodes.csv"
    episodes_json = out_dir / "event_episodes.json"
    track_summary_csv = out_dir / "track_event_summary.csv"

    pd.DataFrame(events, columns=EVENT_COLUMNS).to_csv(events_csv, index=False)
    pd.DataFrame(episodes, columns=EPISODE_COLUMNS).to_csv(episodes_csv, index=False)

    save_frame_events_json(events, frame_events_json, tracking_csv, args)
    save_event_episodes_json(episodes, episodes_json, tracking_csv, args)

    # Track-level summary from episodes, not raw frame events.
    counts = {}
    for ep in episodes:
        for tid in [ep["track_id_1"], ep["track_id_2"]]:
            tid = int(tid)
            if tid not in counts:
                counts[tid] = {
                    "track_id": tid,
                    "num_episodes": 0,
                    "num_potential_conflict": 0,
                    "num_pedestrian_near_vehicle": 0,
                    "num_high": 0,
                    "num_medium": 0,
                    "num_low": 0,
                }

            counts[tid]["num_episodes"] += 1

            if ep["event_type"] == "POTENTIAL_CONFLICT":
                counts[tid]["num_potential_conflict"] += 1
            elif ep["event_type"] == "PEDESTRIAN_NEAR_VEHICLE":
                counts[tid]["num_pedestrian_near_vehicle"] += 1

            if ep["severity"] == "HIGH":
                counts[tid]["num_high"] += 1
            elif ep["severity"] == "MEDIUM":
                counts[tid]["num_medium"] += 1
            elif ep["severity"] == "LOW":
                counts[tid]["num_low"] += 1

    track_rows = sorted(
        counts.values(),
        key=lambda r: (-r["num_high"], -r["num_episodes"], r["track_id"]),
    )
    pd.DataFrame(track_rows).to_csv(track_summary_csv, index=False)

    print("Saved:")
    print(f"  {events_csv}")
    print(f"  {frame_events_json}")
    print(f"  {episodes_csv}")
    print(f"  {episodes_json}")
    print(f"  {track_summary_csv}")
    print()

    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    episodes_df = pd.DataFrame(episodes, columns=EPISODE_COLUMNS)

    print(f"Frame-level events: {len(events_df)}")
    if len(events_df):
        print(events_df.groupby(["event_type", "severity"]).size().to_string())
    print()

    print(f"Event episodes: {len(episodes_df)}")
    if len(episodes_df):
        print(episodes_df.groupby(["event_type", "severity"]).size().to_string())
        print()
        print("Episodes:")
        print(episodes_df[[
            "episode_id",
            "event_type",
            "severity",
            "track_id_1",
            "track_id_2",
            "start_frame_idx",
            "end_frame_idx",
            "num_event_frames",
            "min_distance_m",
            "min_pred_distance_m",
            "frame_of_min_risk",
            "description",
        ]].to_string(index=False))
    else:
        print("No event episodes found. Loosen thresholds or reduce --min-event-frames.")


if __name__ == "__main__":
    main()
