# Runbook

This file contains the practical command sequence for reproducing the current demo checkpoint and for continuing development.

Assumed project root:

```bash
cd /home/cpu-57/heetez/ipt_clean
```

If your path is different, replace it in the commands.

---

## 0. First-time setup after cloning

```bash
cd ipt_clean

git submodule update --init --recursive
```

If the CoopDet3D submodule has not been added yet:

```bash
bash scripts/setup/add_coopdet3d_submodule.sh
```

Expected submodule:

```text
external/coopdet3d
```

Expected fork/branch:

```text
https://github.com/hiteshhedwig/coopdet3d.git
branch: ipt-local-mods
```

---

## 1. Source runtime

Before any CoopDet3D-dependent command:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
cd ../..
```

This is required for:

```text
raw prediction export
CoopDet3D visualization
camera projection using CoopDet3D config
interactive viewer export
BEV + camera replay video rendering
```

---

## 2. Dataset setup

Expected dataset path inside CoopDet3D:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

Create symlink:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```

Example from the development machine:

```bash
bash scripts/setup/setup_data_symlinks.sh /home/cpu-57/heetez/ipt/data/raw/tumtraf_v2x
```

---

## 3. Checkpoint setup

Expected checkpoint:

```text
external/coopdet3d/weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Create folder:

```bash
mkdir -p external/coopdet3d/weights
```

Place checkpoint there.

---

## 4. Generate raw predictions

### 4.1 Train split as eval

This is the prediction export used for the final demo clips.

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh

python tools/export_coop_raw_predictions_train_as_eval.py \
  configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth \
  --out-dir raw_preds_train_as_eval_pretrained

cd ../..
```

Expected output:

```text
external/coopdet3d/raw_preds_train_as_eval_pretrained/
```

Why train-as-eval?

```text
The final selected demo clips are from the train split, but inference/export should use the eval pipeline and GT filename ordering. This avoids dataloader/training-mode mismatch and keeps prediction JSONs aligned with frame IDs.
```

### 4.2 Validation predictions in GT order

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh

python tools/export_coop_raw_predictions_by_gt_order.py \
  configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth \
  --out-dir raw_preds_val_pretrained_gt_order \
  --split val

cd ../..
```

Expected output:

```text
external/coopdet3d/raw_preds_val_pretrained_gt_order/
```

---

## 5. Generate tracking

### 5.1 Flagship clip: 153-233

```bash
bash scripts/pipeline/run_tracking_153_233.sh
```

Equivalent direct command:

```bash
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

Outputs:

```text
outputs/tracking_replay/final_bev_tracking_v1/tracking.mp4
outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv
outputs/tracking_replay/final_bev_tracking_v1/frames/
```

### 5.2 Second clean clip: 075-152

```bash
bash scripts/pipeline/run_tracking_075_152.sh
```

Equivalent direct command:

```bash
python tools/run_tracking_replay_bev.py \
  --dataset-root external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset \
  --pred-root external/coopdet3d/raw_preds_train_as_eval_pretrained \
  --out-dir outputs/tracking_replay/final_bev_tracking_075_152 \
  --split train \
  --score-thr 0.25 \
  --max-age 4 \
  --min-hits 2 \
  --reset-gap-sec 1.5 \
  --start-index 75 \
  --max-frames 78 \
  --save-frames
```

Outputs:

```text
outputs/tracking_replay/final_bev_tracking_075_152/tracking.mp4
outputs/tracking_replay/final_bev_tracking_075_152/diagnostics.csv
outputs/tracking_replay/final_bev_tracking_075_152/frames/
```

---

## 6. Export interactive viewers

Source runtime first:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
cd ../..
```

### 6.1 Viewer for 153-233

```bash
bash scripts/pipeline/export_viewer_153_233.sh
```

Direct command:

```bash
python tools/export_interactive_tracking_viewer.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v5_no_flicker \
  --split train \
  --train-as-eval \
  --start-index 153 \
  --max-frames 81
```

### 6.2 Viewer for 075-152

```bash
bash scripts/pipeline/export_viewer_075_152.sh
```

Direct command:

```bash
python tools/export_interactive_tracking_viewer.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_075_152/diagnostics.csv \
  --out-dir outputs/interactive_viewer/train_as_eval_075_152_v5_no_flicker \
  --split train \
  --train-as-eval \
  --start-index 75 \
  --max-frames 78
```

---

## 7. Event analytics

### 7.1 Pairwise event analytics: 153-233

```bash
bash scripts/pipeline/run_events_153_233.sh
```

Direct command:

```bash
python tools/run_track_event_analytics.py \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/event_analytics/clip_153_233_episodes \
  --prediction-horizon 3.0 \
  --prediction-step 0.5 \
  --ped-vehicle-distance 4.0 \
  --conflict-distance 2.0 \
  --conflict-current-distance-max 10.0 \
  --min-moving-speed 1.2 \
  --min-event-frames 3 \
  --merge-gap-frames 5
```

Outputs:

```text
outputs/event_analytics/clip_153_233_episodes/events.csv
outputs/event_analytics/clip_153_233_episodes/frame_events.json
outputs/event_analytics/clip_153_233_episodes/event_episodes.csv
outputs/event_analytics/clip_153_233_episodes/event_episodes.json
```

### 7.2 Group-crossing event analytics: 153-233

```bash
bash scripts/pipeline/run_group_crossing_153_233.sh
```

Direct command:

```bash
python tools/run_group_crossing_event_analytics.py \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/event_analytics/clip_153_233_group_crossing \
  --cluster-distance 5.0 \
  --min-group-size 2 \
  --prediction-horizon 5.0 \
  --prediction-step 0.5 \
  --same-time-conflict-distance 5.0 \
  --path-conflict-distance 4.0 \
  --max-current-distance 22.0 \
  --min-event-frames 3 \
  --merge-gap-frames 5
```

Outputs:

```text
outputs/event_analytics/clip_153_233_group_crossing/group_crossing_episodes.csv
outputs/event_analytics/clip_153_233_group_crossing/group_crossing_events.json
outputs/event_analytics/clip_153_233_group_crossing/group_crossing_frame_events.csv
```

---

## 8. Inject event overlays into viewers

### 8.1 Pairwise events into 153-233 viewer

```bash
python tools/inject_event_episodes_into_viewer.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v5_no_flicker \
  --episodes-json outputs/event_analytics/clip_153_233_episodes/event_episodes.json \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v6_events
```

### 8.2 Camera-side pairwise event highlights

```bash
python tools/patch_event_camera_highlights.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v6_events \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v7_events_camera_highlight
```

### 8.3 Group crossing overlay

```bash
python tools/inject_group_crossing_into_viewer.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v7_events_camera_highlight \
  --group-json outputs/event_analytics/clip_153_233_group_crossing/group_crossing_events.json \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing \
  --min-severity MEDIUM
```

Stable flagship viewer:

```text
outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing/index.html
```

---

## 9. Render BEV + camera replay videos

```bash
bash scripts/pipeline/render_demo_videos.sh
```

Direct command for 153-233:

```bash
python tools/render_bev_camera_tracking_video.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --dataset-root external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset \
  --bev-frame-dir outputs/tracking_replay/final_bev_tracking_v1/frames \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/combined_viz/final_full_scene_153_233_bev_camera \
  --split train \
  --train-as-eval \
  --start-index 153 \
  --max-frames 81 \
  --fps 10
```

Direct command for 075-152:

```bash
python tools/render_bev_camera_tracking_video.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --dataset-root external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset \
  --bev-frame-dir outputs/tracking_replay/final_bev_tracking_075_152/frames \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_075_152/diagnostics.csv \
  --out-dir outputs/combined_viz/final_full_scene_075_152_bev_camera \
  --split train \
  --train-as-eval \
  --start-index 75 \
  --max-frames 78 \
  --fps 10
```

---

## 10. Serve demo

```bash
bash scripts/pipeline/serve_demo.sh
```

Or manually:

```bash
cd demo/seoul_style_infra_perception_demo
python -m http.server 8000
```

Open:

```text
http://localhost:8000/full_scene_153_233/
http://localhost:8000/full_scene_075_152/
```

---

## 11. Useful validation checks

Check event patches in final HTML:

```bash
grep -R "__EVENT_EPISODE_PATCHED__\|__EVENT_CAMERA_HIGHLIGHT_PATCHED__\|__GROUP_CROSSING_PATCHED__" \
  demo/seoul_style_infra_perception_demo/full_scene_153_233/index.html \
  demo/seoul_style_infra_perception_demo/full_scene_075_152/index.html
```

Expected:

```text
153_233 has EVENT_EPISODE + EVENT_CAMERA + GROUP_CROSSING
075_152 has EVENT_EPISODE + EVENT_CAMERA
```

Check large files:

```bash
find . -type f -size +100M -not -path './.git/*' -printf '%p %k KB\n' | sort -k2 -n
```

Check forbidden files:

```bash
find . \( \
  -name '*.pth' -o \
  -name '*.pt' -o \
  -name '*.ckpt' -o \
  -name '*.pyc' -o \
  -name '*.bak' -o \
  -name '*.bak_*' \
\) -print
```
