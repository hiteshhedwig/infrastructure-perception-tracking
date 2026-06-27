import json
from pathlib import Path
import argparse
import csv


def parse_timestamp(frame_id):
    parts = frame_id.split("_")
    return int(parts[0]) + int(parts[1]) * 1e-9


def count_openlabel_objects(path, score_thr=None):
    if not path.exists():
        return 0

    with open(path, "r") as f:
        data = json.load(f)

    openlabel = data.get("openlabel", data)
    frames = openlabel.get("frames", {})

    count = 0
    scores = []

    for _, frame in frames.items():
        objects = frame.get("objects", {})
        for _, obj in objects.items():
            od = obj.get("object_data", {})
            cuboid = od.get("cuboid", {})
            attrs = cuboid.get("attributes", {})

            score = None
            for item in attrs.get("num", []):
                if item.get("name") == "score":
                    score = float(item.get("val"))
                    break

            if score is not None:
                scores.append(score)

            if score_thr is None:
                count += 1
            else:
                if score is None or score >= score_thr:
                    count += 1

    return count, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset")
    parser.add_argument("--pred-root", default="external/coopdet3d/inference_val_openlabel/openlabel")
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--out-csv", default="outputs/detection_audit_val.csv")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pred_root = Path(args.pred_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    gt_dir = dataset_root / args.split / "labels_point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
    pcd_dir = dataset_root / args.split / "point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"

    gt_files = sorted(gt_dir.glob("*.json"))
    rows = []

    for idx, gt_path in enumerate(gt_files):
        frame_id = gt_path.stem
        pred_path = pred_root / f"{frame_id}.json"
        pcd_path = pcd_dir / f"{frame_id}.pcd"

        gt_count, _ = count_openlabel_objects(gt_path, score_thr=None)
        pred_count_all, scores = count_openlabel_objects(pred_path, score_thr=None) if pred_path.exists() else (0, [])
        pred_count_thr, _ = count_openlabel_objects(pred_path, score_thr=args.score_thr) if pred_path.exists() else (0, [])

        rows.append({
            "frame_idx": idx,
            "frame_id": frame_id,
            "timestamp": parse_timestamp(frame_id),
            "gt_count": gt_count,
            "pred_file_exists": pred_path.exists(),
            "pred_count_all": pred_count_all,
            f"pred_count_score_ge_{args.score_thr}": pred_count_thr,
            "max_score": max(scores) if scores else "",
            "min_score": min(scores) if scores else "",
            "pcd_exists": pcd_path.exists(),
            "pred_path": str(pred_path),
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    pred_files = sum(r["pred_file_exists"] for r in rows)
    gt_positive = sum(r["gt_count"] > 0 for r in rows)
    pred_positive = sum(r[f"pred_count_score_ge_{args.score_thr}"] > 0 for r in rows)
    gt_but_no_pred = [r for r in rows if r["gt_count"] > 0 and r[f"pred_count_score_ge_{args.score_thr}"] == 0]

    print(f"Total frames: {total}")
    print(f"Frames with GT objects: {gt_positive}")
    print(f"Frames with prediction file: {pred_files}")
    print(f"Frames with prediction count >= score_thr: {pred_positive}")
    print(f"Frames with GT but zero predictions at score_thr: {len(gt_but_no_pred)}")
    print(f"CSV saved: {out_csv}")

    print("\nFirst 30 bad frames: GT exists but pred is zero/missing")
    for r in gt_but_no_pred[:30]:
        print(
            f"{r['frame_idx']:03d} | gt={r['gt_count']:02d} | "
            f"pred_file={r['pred_file_exists']} | pred_all={r['pred_count_all']:02d} | "
            f"pred_thr={r[f'pred_count_score_ge_{args.score_thr}']:02d} | {r['frame_id']}"
        )


if __name__ == "__main__":
    main()
