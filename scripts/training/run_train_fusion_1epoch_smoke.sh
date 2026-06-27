#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/external/coopdet3d"
source setup_coopdet3d_runtime.sh
bash run_train_fusion_1epoch_smoke.sh
