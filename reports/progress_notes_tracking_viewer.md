# Infrastructure-Based 3D Perception + Tracking Project Notes

## Current project goal

Build an infrastructure-based 3D perception and tracking system using TUMTraf V2X data.

The pipeline uses roadside / infrastructure LiDAR-camera data to:

1. Run 3D object detection.
2. Track objects across frames.
3. Visualize tracking in BEV.
4. Project tracked objects into camera views.
5. Provide an interactive viewer for debugging and demo purposes.

The goal is not to reproduce the CoopDet3D paper benchmark exactly. The goal is to build a strong perception-engineering project similar to infrastructure autonomy / smart-roadside perception systems.

---

# 1. Important project paths

Main project:

```bash
/home/cpu-57/heetez/ipt
```

CoopDet3D repo:

```bash
/home/cpu-57/heetez/ipt/external/coopdet3d
```

Raw TUMTraf V2X dataset:

```bash
/home/cpu-57/heetez/ipt/data/raw/tumtraf_v2x
```

CoopDet3D dataset symlink:

```bash
/home/cpu-57/heetez/ipt/external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

Processed CoopDet3D dataset:

```bash
/home/cpu-57/heetez/ipt/external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset_processed
```

Official pretrained CoopDet3D checkpoint:

```bash
/home/cpu-57/heetez/ipt/external/coopdet3d/weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Main runtime script:

```bash
/home/cpu-57/heetez/ipt/external/coopdet3d/setup_coopdet3d_runtime.sh
```

---

# 2. Runtime setup

Before running CoopDet3D-related scripts:

```bash
cd /home/cpu-57/heetez/ipt/external/coopdet3d
source setup_coopdet3d_runtime.sh
cd /home/cpu-57/heetez/ipt
```

This sets up:

```text
Python 3.8
PyTorch 1.10.1
CUDA runtime 11.3
Local CUDA toolkit 11.3.1
mmcv-full 1.4.0
mmdet 2.20.0
TorchSparse from source
CoopDet3D local CUDA ops
MPI / Torchpack workaround variables
```

---

# 3. Important lessons learned

## Do not use the OpenLABEL export for tracking

This path gave bad / shifted boxes:

```text
external/coopdet3d/inference_val_openlabel/openlabel
```

The issue was not the model. The issue was the conversion/export path.

Avoid this for tracking.

## Use raw CoopDet3D model boxes instead

Trusted raw prediction folders:

```bash
external/coopdet3d/raw_preds_val_pretrained_gt_order
external/coopdet3d/raw_preds_train_as_eval_pretrained
```

These contain raw `outputs[0]["boxes_3d"]` predictions and align correctly in BEV.

## For train data, use train-as-eval

Normal train dataloader caused problems because of training behavior / augmentation / repeated samples.

Correct approach:

```text
train labels + val/eval pipeline + raw model boxes
```

This produced the trusted train prediction folder:

```bash
external/coopdet3d/raw_preds_train_as_eval_pretrained
```

---

# 4. Best selected clip

The best current clip is:

```text
frames 153-233
start-index = 153
max-frames = 81
duration ≈ 9.9 seconds
```

Start frame:

```text
1688626046_047552884_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered
```

This clip looked clean in:

1. BEV detection alignment.
2. Tracking replay.
3. Camera projection.
4. Interactive viewer.

Use this as the current flagship demo clip.

---

# 5. Find continuous prediction-backed clips

Use this when searching for better clips:

```bash
cd /home/cpu-57/heetez/ipt

python - <<'PY'
import json
from pathlib import Path

split = "train"
dataset_root = Path("external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset")
pred_root = Path("external/coopdet3d/raw_preds_train_as_eval_pretrained")

reset_gap_sec = 1.5
min_pred_boxes = 1

label_dir = (
    dataset_root
    / split
    / "labels_point_clouds"
    / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
)

label_files = sorted(label_dir.glob("*.json"))

def ts_from_frame_id(frame_id: str) -> float:
    parts = frame_id.split("_")
    sec = int(parts[0])
    nsec = int(parts[1])
    return sec + nsec * 1e-9

rows = []

for idx, label_path in enumerate(label_files):
    frame_id = label_path.stem
    ts = ts_from_frame_id(frame_id)

    pred_path = pred_root / f"{frame_id}.json"
    pred_exists = pred_path.exists()
    pred_count = 0

    if pred_exists:
        try:
            d = json.load(open(pred_path))
            pred_count = len(d.get("boxes", []))
        except Exception:
            pred_count = 0

    usable = pred_exists and pred_count >= min_pred_boxes

    rows.append({
        "idx": idx,
        "frame_id": frame_id,
        "timestamp": ts,
        "pred_exists": pred_exists,
        "pred_count": pred_count,
        "usable": usable,
    })

clips = []
cur = []

for r in rows:
    if not r["usable"]:
        if cur:
            clips.append(cur)
            cur = []
        continue

    if not cur:
        cur = [r]
        continue

    dt = r["timestamp"] - cur[-1]["timestamp"]

    if dt <= reset_gap_sec:
        cur.append(r)
    else:
        clips.append(cur)
        cur = [r]

if cur:
    clips.append(cur)

clips = sorted(
    clips,
    key=lambda c: (len(c), c[-1]["timestamp"] - c[0]["timestamp"], sum(x["pred_count"] for x in c) / len(c)),
    reverse=True,
)

print(f"Split: {split}")
print(f"Frames: {len(rows)}")
print(f"Prediction root: {pred_root}")
print(f"Usable frames: {sum(r['usable'] for r in rows)}")
print(f"reset_gap_sec: {reset_gap_sec}")
print()
print("Prediction-backed TRAIN clips:")

for i, c in enumerate(clips[:30]):
    start = c[0]
    end = c[-1]
    duration = end["timestamp"] - start["timestamp"]
    avg_pred = sum(x["pred_count"] for x in c) / len(c)
    max_gap = max(
        [c[j]["timestamp"] - c[j - 1]["timestamp"] for j in range(1, len(c))],
        default=0.0,
    )

    print(
        f"clip {i:02d}: "
        f"frames {start['idx']:03d}-{end['idx']:03d} | "
        f"n={len(c):03d} | "
        f"duration={duration:.2f}s | "
        f"avg_pred={avg_pred:.1f} | "
        f"max_gap={max_gap:.3f}s | "
        f"start_id={start['frame_id']}"
    )
PY
```

The stricter setting:

```text
reset_gap_sec = 1.5
```

gave clean windows like `153-233`.

---

# 6. Generate final BEV tracking video

Current best command:

```bash
cd /home/cpu-57/heetez/ipt

rm -rf outputs/tracking_replay/final_bev_tracking_v1

python tools/run_tracking_replay_bev.py \
  --dataset-root external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset \
  --pred-root external/coopdet3d/raw_preds_train_as_eval_pretrained \
  --out-dir outputs/tracking_replay/final_bev_tracking_v1 \
  --split train \
  --score-thr 0.25 \
  --max-age 4 \
  --min-hits 2 \
  --reset-gap-sec 1.5 \
  --start-index 153 \
  --max-frames 81 \
  --save-frames
```

Important output:

```bash
outputs/tracking_replay/final_bev_tracking_v1/tracking.mp4
outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv
outputs/tracking_replay/final_bev_tracking_v1/frames/
```

The diagnostics CSV now includes:

```text
frame_idx
frame_id
timestamp
track_id
class_name
x, y, z
l, w, h, yaw
vx, vy
speed_mps
age
hits
time_since_update
last_detection_score
last_match_distance
num_detections
has_pred_file
```

This CSV is important because it is used for the combined video and the interactive viewer.

---

# 7. Generate BEV + camera combined video

This creates a normal MP4 video:

```text
Left: BEV tracking
Right: four camera views
```

Command:

```bash
cd /home/cpu-57/heetez/ipt/external/coopdet3d
source setup_coopdet3d_runtime.sh
cd /home/cpu-57/heetez/ipt

rm -rf outputs/combined_viz/bev_plus_camera_train_as_eval_153_233_v2_bold_labels

python tools/render_bev_camera_tracking_video.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --dataset-root external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset \
  --bev-frame-dir outputs/tracking_replay/final_bev_tracking_v1/frames \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/combined_viz/bev_plus_camera_train_as_eval_153_233_v2_bold_labels \
  --split train \
  --train-as-eval \
  --start-index 153 \
  --max-frames 81 \
  --fps 10
```

Output:

```bash
outputs/combined_viz/bev_plus_camera_train_as_eval_153_233_v2_bold_labels/bev_plus_cameras.mp4
```

Open:

```bash
xdg-open outputs/combined_viz/bev_plus_camera_train_as_eval_153_233_v2_bold_labels
```

---

# 8. Generate interactive HTML tracking viewer

This is the best current demo/debug tool.

It creates a portable static viewer:

```text
index.html
metadata.json
assets/cameras/...
```

Features:

1. Select all tracks or one track.
2. Play/pause.
3. Frame slider.
4. BEV canvas.
5. Four camera views.
6. Projected 3D boxes in camera.
7. Track details: class, speed, score, frame range.
8. Toggle camera boxes.
9. Toggle BEV trails.

Command:

```bash
cd /home/cpu-57/heetez/ipt/external/coopdet3d
source setup_coopdet3d_runtime.sh
cd /home/cpu-57/heetez/ipt

rm -rf outputs/interactive_viewer/train_as_eval_153_233_v2_scaled_projection

python tools/export_interactive_tracking_viewer.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v2_scaled_projection \
  --split train \
  --train-as-eval \
  --start-index 153 \
  --max-frames 81
```

Open directly:

```bash
xdg-open outputs/interactive_viewer/train_as_eval_153_233_v2_scaled_projection/index.html
```

If local browser loading has issues:

```bash
cd outputs/interactive_viewer/train_as_eval_153_233_v2_scaled_projection
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

This viewer is currently one of the strongest project artifacts.

---

# 9. Current trusted scripts

Main custom scripts created / modified:

```bash
tools/run_tracking_replay_bev.py
tools/render_bev_camera_tracking_video.py
tools/export_interactive_tracking_viewer.py
tools/export_coop_raw_predictions_train_as_eval.py
tools/compare_coopdet3d_gt_pred_bev.py
```

Important behavior:

## `run_tracking_replay_bev.py`

Uses raw predictions and tracks objects in BEV.

Outputs:

```text
tracking.mp4
diagnostics.csv
frames/
```

It now saves `l,w,h,yaw`, which are required for camera projection.

## `render_bev_camera_tracking_video.py`

Creates MP4 combined visualization:

```text
BEV + 4 camera panels
```

Good for quick portfolio video.

## `export_interactive_tracking_viewer.py`

Creates a static HTML demo:

```text
Interactive track selector
BEV canvas
Camera panels
Projected boxes
Metadata panel
```

Good for debugging and project showcase.

---

# 10. Known issues / decisions

## Minor camera projection offset

Some camera boxes are slightly shifted for moving objects.

Decision:

```text
Ignore for now.
```

Reason:

1. GT camera projection is okay.
2. Pred camera projection is mostly okay.
3. BEV is the source of truth.
4. The remaining offset is minor and likely related to timestamp / sensor sync / motion.

## Yaw jitter

Detector yaw can look jittery in BEV.

Decision:

```text
Do not override model yaw.
```

We tried velocity-based yaw correction but it looked too aggressive.

Final decision:

```text
Box yaw = detector yaw.
Motion direction = future dots / trails.
```

This is cleaner and more honest.

## Tracker quality

Tracker is acceptable for visualization, but not final.

Later improvements:

1. Better association logic.
2. BEV IoU + center-distance matching.
3. Class-compatible matching.
4. Smoother velocity estimation.
5. Better track lifecycle logic.
6. Track metrics against GT track IDs if needed.

---

# 11. Current milestone summary

We completed:

```text
Official pretrained CoopDet3D inference
→ raw prediction export
→ prediction/GT alignment debugging
→ train-as-eval inference path
→ BEV online tracking replay
→ camera projection validation
→ combined BEV + camera video
→ static interactive HTML viewer
```

Main achievement:

```text
We now have a working infrastructure-based 3D perception and tracking demo using TUMTraf V2X, pretrained CoopDet3D detections, BEV tracking, camera projection, and an interactive track viewer.
```

---

# 12. Best outputs to showcase

BEV tracking video:

```bash
outputs/tracking_replay/final_bev_tracking_v1/tracking.mp4
```

Combined BEV + camera video:

```bash
outputs/combined_viz/bev_plus_camera_train_as_eval_153_233_v2_bold_labels/bev_plus_cameras.mp4
```

Interactive viewer:

```bash
outputs/interactive_viewer/train_as_eval_153_233_v2_scaled_projection/index.html
```

These three are the current strongest deliverables.

---

# 13. Next recommended step

Next step should be:

```text
Add simple event/risk detection on top of tracks.
```

Possible event flags:

1. Stopped vehicle.
2. Fast-moving object.
3. Pedestrian near vehicle.
4. Object entering conflict zone.
5. Wrong-way movement.
6. Near-collision / close approach.

Add these to:

```text
diagnostics.csv
metadata.json
interactive viewer details panel
BEV overlays
```

This will make the project more than detection/tracking — it becomes infrastructure traffic intelligence.
