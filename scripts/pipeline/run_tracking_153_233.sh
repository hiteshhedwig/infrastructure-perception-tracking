#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
python tools/run_tracking_replay_bev.py \
  --dataset-root external/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset \
  --pred-root external/coopdet3d/raw_preds_train_as_eval_pretrained \
  --out-dir outputs/tracking_replay/final_bev_tracking_v1 \
  --split train --score-thr 0.25 --max-age 4 --min-hits 2 --reset-gap-sec 1.5 \
  --start-index 153 --max-frames 81 --save-frames
