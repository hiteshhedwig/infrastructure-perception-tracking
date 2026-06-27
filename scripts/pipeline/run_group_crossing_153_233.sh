#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
python tools/run_group_crossing_event_analytics.py \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/event_analytics/clip_153_233_group_crossing \
  --cluster-distance 5.0 --min-group-size 2 --prediction-horizon 5.0 --prediction-step 0.5 \
  --same-time-conflict-distance 5.0 --path-conflict-distance 4.0 --max-current-distance 22.0 \
  --min-event-frames 3 --merge-gap-frames 5
