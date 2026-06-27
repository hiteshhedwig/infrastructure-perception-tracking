# Setup

1. Initialize Git.
2. Add CoopDet3D submodule.
3. Setup dataset symlink.
4. Place checkpoint locally.
5. Source CoopDet3D runtime before model-dependent commands.

```bash
git init
bash scripts/setup/add_coopdet3d_submodule.sh
bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x
cd external/coopdet3d
source setup_coopdet3d_runtime.sh
cd ../..
```
