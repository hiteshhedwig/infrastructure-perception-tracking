# Model Setup

Large checkpoint files are not committed to normal Git.

---

## Current pretrained checkpoint

Checkpoint name:

```text
coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Expected local path:

```text
external/coopdet3d/weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Create folder:

```bash
mkdir -p external/coopdet3d/weights
```

Then place the checkpoint there.

---

## Config used

```text
external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml
```

---

## Model storage policy

Do not commit these directly to normal Git:

```text
*.pth
*.pt
*.ckpt
```

Use one of:

```text
GitHub Release assets
Git LFS
Hugging Face model repo
Google Drive
local checkpoint path
```

---

## Future training result organization

For future ablations and trained models:

```text
models/
  checkpoints/
    experiment_name/
      README.md
      config.yaml
      metrics.json
      checkpoint_link.txt
```

Example:

```text
models/checkpoints/lidar_pointpillars_20epoch/
  README.md
  config.yaml
  metrics.json
  checkpoint_link.txt
```

---

## Smoke test

After placing checkpoint and dataset:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh

python tools/collect_env.py
```

Then run raw prediction export on the relevant split. See:

```text
docs/inference_generation.md
```
