# Troubleshooting

## 1. Camera boxes look shifted

Use the stable viewer export code that uses MMDetection3D `LiDARInstance3DBoxes` corner conventions.

Do not use the old custom corner function.

Stable viewer script:

```text
tools/export_interactive_tracking_viewer.py
```

---

## 2. Tracking yaw jitters

Do not force yaw from velocity direction.

The stable decision is:

```text
track yaw = detector/model yaw
motion cues = trails/future dots
```

---

## 3. OpenLABEL predictions look wrong

Do not use old OpenLABEL outputs as tracking input.

Avoid:

```text
external/coopdet3d/inference_val_openlabel/
external/coopdet3d/inference_train_openlabel_pretrained/
```

Use:

```text
external/coopdet3d/raw_preds_train_as_eval_pretrained/
external/coopdet3d/raw_preds_val_pretrained_gt_order/
```

---

## 4. Missing CoopDet3D

Run:

```bash
git submodule update --init --recursive
```

Or:

```bash
bash scripts/setup/add_coopdet3d_submodule.sh
```

---

## 5. Missing dataset

Expected:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

Fix:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```

---

## 6. Missing checkpoint

Expected:

```text
external/coopdet3d/weights/coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth
```

Fix:

```bash
mkdir -p external/coopdet3d/weights
```

Place checkpoint there.

---

## 7. CUDA/MMCV/TorchSparse import errors

Source runtime:

```bash
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
cd ../..
```

If still broken, verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import mmdet; print(mmdet.__version__)"
python -c "import torchsparse; print('torchsparse ok')"
```

---

## 8. Demo HTML opens but images do not load

Serve with HTTP server. Do not open `index.html` directly from file explorer.

```bash
cd demo/seoul_style_infra_perception_demo
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/full_scene_153_233/
```

---

## 9. Repository accidentally contains huge files

Check:

```bash
find . -type f -size +100M -not -path './.git/*' -printf '%p %k KB\n' | sort -k2 -n
```

Remove model/data/output files from Git and add them to `.gitignore`.

---

## 10. Accidentally committed generated outputs

Generated outputs should stay under:

```text
outputs/
```

and should be ignored, except:

```text
outputs/README.md
outputs/.gitkeep
```

Final demo artifacts should live under:

```text
demo/
```

not `outputs/`.
