# Dataset Setup

The full TUMTraf-V2X dataset is not committed to this repository.

---

## Expected local layout

Project-side layout:

```text
data/
  raw/
    tumtraf_v2x/
      train/
      val/
      test/

  processed/
    frame_indices/
      train_frames.csv
      val_frames.csv
      test_frames.csv
```

CoopDet3D expected layout:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

---

## Create symlink

Use the helper script:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```

Example:

```bash
bash scripts/setup/setup_data_symlinks.sh /home/cpu-57/heetez/ipt/data/raw/tumtraf_v2x
```

This creates:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset -> /absolute/path/to/tumtraf_v2x
```

---

## Frame indices

Small frame-index CSVs are kept in Git:

```text
data/processed/frame_indices/train_frames.csv
data/processed/frame_indices/val_frames.csv
data/processed/frame_indices/test_frames.csv
```

These are useful for quick inspection, clip search, and reproducibility.

---

## What not to commit

Do not commit:

```text
data/raw/tumtraf_v2x/train/
data/raw/tumtraf_v2x/val/
data/raw/tumtraf_v2x/test/
external/coopdet3d/data/
point cloud files
camera image files
full labels
processed dataset folders
```

---

## Useful dataset inspection commands

```bash
tree -h -L 3 data
tree -h -L 3 external/coopdet3d/data
```

Count labels:

```bash
find external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset/train -name '*.json' | wc -l
find external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset/val -name '*.json' | wc -l
```

Build frame index if needed:

```bash
python tools/build_frame_index.py
```

Dataset statistics:

```bash
python tools/dataset_statistics.py
```
