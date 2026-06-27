#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/setup/setup_data_symlinks.sh /absolute/path/to/tumtraf_v2x"
  exit 1
fi
DATASET_SRC="$1"
mkdir -p external/coopdet3d/data
rm -f external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
ln -s "$DATASET_SRC" external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset
echo "Symlink created."
