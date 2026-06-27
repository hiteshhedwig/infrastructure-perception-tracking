# Inference and Raw Prediction Generation

This document explains how to generate model predictions used by the tracking and visualization pipeline.

---

## Why raw prediction export is used

For this project, tracking should use raw CoopDet3D model boxes, not the earlier OpenLABEL export.

Trusted source:

```python
outputs[0]["boxes_3d"]
```

Reason:

```text
The earlier OpenLABEL inference outputs were useful during early testing, but the box convention was not reliable enough for downstream tracking. The stable pipeline exports raw model boxes in ground-truth filename order and then consumes those JSONs directly.
```

---

## Required runtime

Always source CoopDet3D runtime before inference/export:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
```

Expected environment used during development:

```text
Python 3.8
PyTorch 1.10.1
CUDA 11.3
mmcv-full 1.4.0
mmdet 2.20.0
TorchSparse built from source
CoopDet3D CUDA ops compiled
```

---

## Required files

Config:

```text
external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml
```

Checkpoint:

```text
external/coopdet3d/weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Dataset:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

---

## Generate train split predictions using eval pipeline

This is the prediction folder used by the final demo clips.

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

Each prediction JSON is named by frame ID:

```text
1688626046_047552884_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json
```

---

## Generate validation predictions in GT order

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

## Prediction JSON schema

The raw prediction JSON files are expected to contain a list of boxes with:

```text
class_name
score
x, y, z
l, w, h
yaw
```

The downstream tracker reads these files by matching the frame ID from the TUMTraf label filename.

---

## Final trusted prediction folders

```text
external/coopdet3d/raw_preds_train_as_eval_pretrained
external/coopdet3d/raw_preds_val_pretrained_gt_order
```

---

## Do not use these as tracking source

Avoid:

```text
external/coopdet3d/inference_val_openlabel/
external/coopdet3d/inference_train_openlabel_pretrained/
```

These folders may be kept locally for comparison, but they are not the stable tracking source.

---

## Quick sanity check

Count prediction files:

```bash
find external/coopdet3d/raw_preds_train_as_eval_pretrained -name '*.json' | wc -l
find external/coopdet3d/raw_preds_val_pretrained_gt_order -name '*.json' | wc -l
```

Inspect one prediction:

```bash
python - <<'PY'
import json
from pathlib import Path

p = sorted(Path("external/coopdet3d/raw_preds_train_as_eval_pretrained").glob("*.json"))[0]
d = json.load(open(p))
print(p.name)
print(d.keys())
print("num boxes:", len(d.get("boxes", [])))
print(d["boxes"][0] if d.get("boxes") else "no boxes")
PY
```

---

## Common failure modes

### Missing checkpoint

Error:

```text
No such file or directory: weights/...pth
```

Fix:

```bash
mkdir -p external/coopdet3d/weights
```

Then place the checkpoint at the expected path.

### Missing dataset

Error usually mentions missing infos, point cloud files, images, or labels.

Fix:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```

### Import/CUDA/MMCV errors

Fix by sourcing runtime:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
```

If still broken, rebuild TorchSparse and CoopDet3D ops in the matching conda/CUDA environment.
