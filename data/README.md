# Data

Large datasets are not committed.

Expected local structure:

```text
data/raw/tumtraf_v2x/{train,val,test}
data/processed/frame_indices/{train_frames.csv,val_frames.csv,test_frames.csv}
data/predictions/
```

CoopDet3D expects a dataset path like:

```text
external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
```

Create a symlink using:

```bash
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
```
