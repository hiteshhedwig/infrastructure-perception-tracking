#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
python tools/run_track_event_analytics.py \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/event_analytics/clip_153_233_episodes \
  --prediction-horizon 3.0 --prediction-step 0.5 --ped-vehicle-distance 4.0 \
  --conflict-distance 2.0 --conflict-current-distance-max 10.0 \
  --min-moving-speed 1.2 --min-event-frames 3 --merge-gap-frames 5
