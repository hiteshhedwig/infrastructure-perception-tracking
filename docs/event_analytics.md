# Event Analytics

This project currently supports three event/risk concepts.

---

## Event types

```text
PEDESTRIAN_NEAR_VEHICLE
POTENTIAL_CONFLICT
CROSSING_GROUP_VEHICLE_RISK
```

The first two are pairwise track events. The third is a group-level pedestrian crossing risk event.

---

## Pairwise event analytics

Script:

```text
tools/run_track_event_analytics.py
```

Input:

```text
tracking diagnostics CSV
```

Output:

```text
events.csv
frame_events.json
event_episodes.csv
event_episodes.json
track_event_summary.csv
```

Command for 153-233:

```bash
python tools/run_track_event_analytics.py \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/event_analytics/clip_153_233_episodes \
  --prediction-horizon 3.0 \
  --prediction-step 0.5 \
  --ped-vehicle-distance 4.0 \
  --conflict-distance 2.0 \
  --conflict-current-distance-max 10.0 \
  --min-moving-speed 1.2 \
  --min-event-frames 3 \
  --merge-gap-frames 5
```

---

## Group crossing event analytics

Script:

```text
tools/run_group_crossing_event_analytics.py
```

Input:

```text
tracking diagnostics CSV
```

Output:

```text
group_crossing_episodes.csv
group_crossing_events.json
group_crossing_frame_events.csv
```

Command for 153-233:

```bash
python tools/run_group_crossing_event_analytics.py \
  --tracking-csv outputs/tracking_replay/final_bev_tracking_v1/diagnostics.csv \
  --out-dir outputs/event_analytics/clip_153_233_group_crossing \
  --cluster-distance 5.0 \
  --min-group-size 2 \
  --prediction-horizon 5.0 \
  --prediction-step 0.5 \
  --same-time-conflict-distance 5.0 \
  --path-conflict-distance 4.0 \
  --max-current-distance 22.0 \
  --min-event-frames 3 \
  --merge-gap-frames 5
```

---

## Viewer overlay injection

Pairwise event panel:

```bash
python tools/inject_event_episodes_into_viewer.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v5_no_flicker \
  --episodes-json outputs/event_analytics/clip_153_233_episodes/event_episodes.json \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v6_events
```

Camera event highlights:

```bash
python tools/patch_event_camera_highlights.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v6_events \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v7_events_camera_highlight
```

Group crossing overlay:

```bash
python tools/inject_group_crossing_into_viewer.py \
  --viewer-dir outputs/interactive_viewer/train_as_eval_153_233_v7_events_camera_highlight \
  --group-json outputs/event_analytics/clip_153_233_group_crossing/group_crossing_events.json \
  --out-dir outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing \
  --min-severity MEDIUM
```

---

## Confirm overlay markers

```bash
grep -R "__EVENT_EPISODE_PATCHED__\|__EVENT_CAMERA_HIGHLIGHT_PATCHED__\|__GROUP_CROSSING_PATCHED__" \
  outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing/index.html
```

Expected:

```text
__EVENT_EPISODE_PATCHED__
__EVENT_CAMERA_HIGHLIGHT_PATCHED__
__GROUP_CROSSING_PATCHED__
```

---

## Interpretation

For V1, event analytics are demo-grade and interpretable. They are not yet a validated safety system.

Use them as:

```text
track-derived risk cues
event-level visualization
portfolio/research demo logic
starting point for future ablation studies
```

Do not present them as production-certified collision prediction.
