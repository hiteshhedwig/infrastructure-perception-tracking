# CHECKPOINT_V1_SEOUL_STYLE_INFRA_PERCEPTION_DEMO

This checkpoint marks the first stable demo package for the infrastructure perception project.

---

## Stable scenes

```text
full_scene_153_233
full_scene_075_152
```

---

## Scene 153-233

Status:

```text
Flagship full-scene demo
```

Includes:

```text
3D detections
online tracking
BEV point-cloud background
camera projection
event panel
camera-side event highlights
pedestrian group crossing risk
vehicle/group future path visualization
```

Stable source viewer:

```text
outputs/interactive_viewer/train_as_eval_153_233_v8_group_crossing/index.html
```

Final demo path:

```text
demo/seoul_style_infra_perception_demo/full_scene_153_233/index.html
```

---

## Scene 075-152

Status:

```text
Second clean full-scene demo
```

Includes:

```text
3D detections
online tracking
BEV point-cloud background
camera projection
event panel
camera-side event highlights
```

Stable source viewer:

```text
outputs/interactive_viewer/train_as_eval_075_152_v7_events_camera_highlight/index.html
```

Final demo path:

```text
demo/seoul_style_infra_perception_demo/full_scene_075_152/index.html
```

---

## Replay videos

```text
demo/seoul_style_infra_perception_demo/videos/full_scene_153_233_bev_plus_cameras.mp4
demo/seoul_style_infra_perception_demo/videos/full_scene_075_152_bev_plus_cameras.mp4
```

---

## Important decisions

```text
Use raw CoopDet3D boxes, not old OpenLABEL tracking source.
Use train-as-eval prediction export for train clips.
Keep detector yaw for track orientation.
Use trails/future dots for motion cues.
Use 153-233 as flagship.
Use 075-152 as second clean clip.
Avoid 630-716 as final demo because of visual/projection issues.
```
