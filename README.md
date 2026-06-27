# IPT: Infrastructure Perception, Tracking, and Event Analytics

IPT is a development repository for an infrastructure-based 3D perception stack.

The current checkpoint demonstrates a roadside autonomy workflow:

```text
TUMTraf-V2X roadside / cooperative sensor data
→ CoopDet3D 3D detection
→ raw prediction export
→ online BEV tracking
→ multi-camera 3D box projection
→ interactive HTML visualization
→ traffic event and risk analytics
→ final demo packaging
```

This repository is intended to stay small enough for GitHub while still being useful for future research work such as custom training, ablation studies, new model checkpoints, new risk logic, and production-style visualization.

---

## Current checkpoint

```text
CHECKPOINT_V1_STYLE_INFRA_PERCEPTION_DEMO
```

Stable demo:

```text
demo/style_infra_perception_demo/
```

Main demo scenes:

```text
full_scene_153_233/   # flagship full-scene demo with group-crossing risk
full_scene_075_152/   # second clean full-scene tracking/event demo
videos/               # BEV + camera replay MP4s
```

---

## Repository layout

```text
ipt_clean/
  README.md
  requirements.txt
  .gitignore

  configs/
    tumtraf_v2x.yaml

  data/
    README.md
    raw/
      tumtraf_v2x/
    processed/
      frame_indices/
      tumtraf_v2x/
    predictions/

  docs/
    dataset_setup.md
    model_setup.md
    inference_generation.md
    pipeline.md
    event_analytics.md
    demo.md
    runbook.md
    troubleshooting.md
    project_structure.md
    checkpoint_v1_summary.md

  external/
    README.md
    coopdet3d/                     # Git submodule: hiteshhedwig/coopdet3d, branch ipt-local-mods

  models/
    README.md
    model_zoo.md
    checkpoints/

  src/
    data/
    detection/
    tracking/
    events/
    visualization/
    utils/

  tools/
    run_tracking_replay_bev.py
    render_bev_camera_tracking_video.py
    export_interactive_tracking_viewer.py
    run_track_event_analytics.py
    run_group_crossing_event_analytics.py
    inject_event_episodes_into_viewer.py
    patch_event_camera_highlights.py
    inject_group_crossing_into_viewer.py
    compare_coopdet3d_gt_pred_bev.py
    audit_val_detection_coverage.py
    dataset_statistics.py
    build_frame_index.py
    test_tumtraf_loader.py

  tools_debug/
    debug and inspection utilities

  scripts/
    setup/
    pipeline/
    training/

  demo/
    style_infra_perception_demo/

  outputs/
    local generated outputs, ignored by Git
```

---

## External dependency: CoopDet3D fork

This project uses a forked CoopDet3D branch as a Git submodule:

```text
https://github.com/hiteshhedwig/coopdet3d.git
branch: ipt-local-mods
```

That fork contains the custom CoopDet3D-side utilities needed by this project, especially:

```text
setup_coopdet3d_runtime.sh
tools/export_coop_raw_predictions_train_as_eval.py
tools/export_coop_raw_predictions_by_gt_order.py
tools/export_coop_raw_predictions.py
run_train_lidar_smoke.sh
run_train_lidar_20epoch.sh
run_train_fusion_1epoch_smoke.sh
run_train_fusion_8epoch_paperstyle.sh
```

After cloning this repo, add/update the submodule:

```bash
git submodule update --init --recursive
```

If the submodule has not been added yet:

```bash
bash scripts/setup/add_coopdet3d_submodule.sh
```

---

## What is included and what is not included

Included:

```text
Project scripts
Tracking pipeline
Event analytics
Interactive viewer export code
Demo HTML assets
Demo replay videos
Small frame-index metadata
Documentation and run commands
```

Not included:

```text
Full TUMTraf-V2X dataset
Large model checkpoints
Generated raw prediction folders
Training work_dirs/runs
Old experiment outputs
Python cache files
Backup files
```

The excluded items should stay local or be distributed using GitHub Releases, Git LFS, Hugging Face, Google Drive, or another artifact store.

---

## Setup overview

### 1. Clone the repo

```bash
git clone <YOUR_IPT_REPO_URL> ipt_clean
cd ipt_clean
```

### 2. Initialize CoopDet3D submodule

```bash
git submodule update --init --recursive
```

Or, if starting from a folder without the submodule:

```bash
bash scripts/setup/add_coopdet3d_submodule.sh
```

### 3. Prepare the CoopDet3D runtime

The working runtime used during development was:

```text
Python 3.8
PyTorch 1.10.1
CUDA 11.3
mmcv-full 1.4.0
mmdet 2.20.0
TorchSparse built from source
CoopDet3D CUDA ops compiled
```

Before running CoopDet3D-dependent scripts:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
cd ../..
```

### 4. Prepare dataset paths

The full dataset is not committed. Expected local path:

```text
data/raw/tumtraf_v2x/
```

Expected CoopDet3D path:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

Create a symlink:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```

### 5. Place checkpoint

Expected checkpoint path:

```text
external/coopdet3d/weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Create folder:

```bash
mkdir -p external/coopdet3d/weights
```

Then place the checkpoint there.

---

## Running the existing demo

If the demo folder is included:

```bash
cd demo/style_infra_perception_demo
python -m http.server 8000
```

Open:

```text
http://localhost:8000/full_scene_153_233/
http://localhost:8000/full_scene_075_152/
```

Or use:

```bash
bash scripts/pipeline/serve_demo.sh
```

---

## Regenerating the pipeline

The short version:

```bash
cd ipt_clean

# 1. Source runtime
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
cd ../..

# 2. Generate raw predictions if not already available
cd external/coopdet3d
python tools/export_coop_raw_predictions_train_as_eval.py \
  configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth \
  --out-dir raw_preds_train_as_eval_pretrained
cd ../..

# 3. Tracking
bash scripts/pipeline/run_tracking_153_233.sh
bash scripts/pipeline/run_tracking_075_152.sh

# 4. Viewer export
bash scripts/pipeline/export_viewer_153_233.sh
bash scripts/pipeline/export_viewer_075_152.sh

# 5. Event analytics
bash scripts/pipeline/run_events_153_233.sh
bash scripts/pipeline/run_group_crossing_153_233.sh

# 6. Replay videos
bash scripts/pipeline/render_demo_videos.sh
```

See detailed docs:

```text
docs/inference_generation.md
docs/pipeline.md
docs/event_analytics.md
docs/runbook.md
```

---

## Important implementation decisions

### Use raw CoopDet3D boxes, not old OpenLABEL output

The old OpenLABEL export path was useful for early visualization but should not be used as the tracking source. It had box convention issues for this pipeline. The stable tracking pipeline uses raw CoopDet3D model boxes from:

```text
outputs[0]["boxes_3d"]
```

saved in ground-truth filename order.

Trusted prediction folders:

```text
external/coopdet3d/raw_preds_train_as_eval_pretrained
external/coopdet3d/raw_preds_val_pretrained_gt_order
```

### Track yaw should remain detector yaw

Do not override track yaw with velocity yaw. Velocity yaw caused visual jitter. Motion is shown using trails and future dots instead.

### Final demo clips

```text
153-233    flagship clip
075-152    second clean clip
```

Avoid using the old `630-716` candidate as a main demo because it had visual/projection quality issues.

---

## Future work

Suggested next development steps:

```text
1. Refactor stable tool scripts into src/ package modules.
2. Add experiment configs for training and ablation studies.
3. Add model registry entries for custom trained checkpoints.
4. Add a structured results table for detection/tracking/event metrics.
5. Add a clean one-command build script for full demo regeneration.
6. Add a GitHub Release artifact for the full demo folder if repository size becomes too large.
```

---

## Citation and license notes

This project depends on external work:

```text
TUMTraf-V2X dataset
CoopDet3D
MMDetection3D / MMDetection / MMCV
TorchSparse
```

Keep their original licenses and citations in the final public repository. Do not claim external code, datasets, or checkpoints as original work.
