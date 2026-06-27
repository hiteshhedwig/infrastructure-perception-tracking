# Project Structure

This repository is organized as a clean development repo, not just a demo dump.

---

## Root folders

```text
configs/        Project-level configs.
data/           Dataset structure and small metadata only.
docs/           Documentation and command guides.
external/       Git submodules and third-party dependency notes.
models/         Checkpoint instructions and model registry.
src/            Future importable package code.
tools/          Core runnable scripts.
tools_debug/    Debugging and inspection utilities.
scripts/        Reproducible setup, pipeline, and training wrappers.
reports/        Progress notes and checkpoint summaries.
demo/           Final static demo assets.
outputs/        Local generated outputs, ignored by Git.
```

---

## tools/

Production-use scripts:

```text
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
```

---

## tools_debug/

Debug and exploration scripts. These are useful for development, but are separated from the main pipeline.

Examples:

```text
debug_pred_transform_candidates.py
debug_project_points_all_cameras.py
inspect_openlabel.py
visualize_one_bev_overlay.py
create_frame_contact_sheet.py
```

---

## external/

Expected:

```text
external/
  coopdet3d/
```

`external/coopdet3d` should be a Git submodule pointing to:

```text
https://github.com/hiteshhedwig/coopdet3d.git
branch: ipt-local-mods
```

Do not commit generated artifacts inside the submodule.

---

## data/

The dataset itself is not committed.

Committed:

```text
data/README.md
data/processed/frame_indices/*.csv
.gitkeep files
```

Ignored:

```text
data/raw/tumtraf_v2x/**
data/processed/tumtraf_v2x/**
data/predictions/**
```

---

## models/

Model files are not committed directly.

Committed:

```text
models/README.md
models/model_zoo.md
models/checkpoints/.gitkeep
```

Ignored:

```text
*.pth
*.pt
*.ckpt
```

---

## demo/

Final static demo package.

This can be committed if repository size is acceptable, or moved to GitHub Releases if it becomes too large.

---

## outputs/

Local generated outputs.

Ignored by Git:

```text
outputs/tracking_replay/
outputs/interactive_viewer/
outputs/combined_viz/
outputs/event_analytics/
outputs/final_demo/
```
