# Pipeline Documentation

This document describes the stable V1 pipeline.

---

## Pipeline stages

```text
1. Dataset setup
2. CoopDet3D raw prediction export
3. BEV online tracking
4. Interactive viewer export
5. Pairwise event analytics
6. Group crossing event analytics
7. Event overlay injection
8. BEV + camera replay rendering
9. Final demo packaging
```

---

## Stage 1: Dataset setup

Expected dataset path:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

Create symlink:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```

---

## Stage 2: Raw prediction export

Generate train-as-eval predictions:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh

python tools/export_coop_raw_predictions_train_as_eval.py \
  configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth \
  --out-dir raw_preds_train_as_eval_pretrained

cd ../..
```

Output:

```text
external/coopdet3d/raw_preds_train_as_eval_pretrained/
```

---

## Stage 3: Tracking

Flagship clip:

```bash
bash scripts/pipeline/run_tracking_153_233.sh
```

Second clip:

```bash
bash scripts/pipeline/run_tracking_075_152.sh
```

Tracking outputs:

```text
outputs/tracking_replay/<run_name>/tracking.mp4
outputs/tracking_replay/<run_name>/diagnostics.csv
outputs/tracking_replay/<run_name>/frames/
```

The diagnostics CSV is the main contract between tracking, event analytics, viewer export, and video rendering.

Important columns:

```text
frame_idx
frame_id
timestamp
track_id
class_name
x, y, z
l, w, h
yaw
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

---

## Stage 4: Viewer export

```bash
bash scripts/pipeline/export_viewer_153_233.sh
bash scripts/pipeline/export_viewer_075_152.sh
```

Viewer output:

```text
outputs/interactive_viewer/<viewer_name>/
  index.html
  metadata.json
  assets/
    bev/
    cameras/
```

Viewer features:

```text
BEV point-cloud background
camera panels
3D boxes projected into cameras
track dropdown
track trails
selected-track playback
frame slider
details panel
```

---

## Stage 5: Event analytics

Pairwise events:

```bash
bash scripts/pipeline/run_events_153_233.sh
```

Group crossing risk:

```bash
bash scripts/pipeline/run_group_crossing_153_233.sh
```

---

## Stage 6: Event overlay injection

Pairwise event panel:

```bash
python tools/inject_event_episodes_into_viewer.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v5_no_flicker \
  --episodes-json outputs/event_analytics/clip_153_233_episodes/event_episodes.json \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v6_events
```

Camera highlights:

```bash
python tools/patch_event_camera_highlights.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v6_events \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v7_events_camera_highlight
```

Group crossing overlay:

```bash
python tools/inject_group_crossing_into_viewer.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v7_events_camera_highlight \
  --group-json outputs/event_analytics/clip_153_233_group_crossing/group_crossing_events.json \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing \
  --min-severity MEDIUM
```

Final flagship viewer:

```text
outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing/index.html
```

---

## Stage 7: Replay videos

```bash
bash scripts/pipeline/render_demo_videos.sh
```

Outputs:

```text
outputs/combined_viz/final_full_scene_153_233_bev_camera/bev_plus_cameras.mp4
outputs/combined_viz/final_full_scene_075_152_bev_camera/bev_plus_cameras.mp4
```

---

## Stage 8: Final demo package

Final package structure:

```text
demo/style_infra_perception_demo/
  README.md
  start_demo_server.sh
  full_scene_153_233/
  full_scene_075_152/
  videos/
```

Serve:

```bash
bash scripts/pipeline/serve_demo.sh
```
