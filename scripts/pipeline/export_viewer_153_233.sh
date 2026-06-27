#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
python tools/export_interactive_tracking_viewer.py \
  --config external/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v5_no_flicker \
  --split train --train-as-eval --start-index 153 --max-frames 81
