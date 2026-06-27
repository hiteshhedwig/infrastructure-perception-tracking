# Demo

The final demo is a static HTML + assets + MP4 package.

---

## Demo path

```text
demo/style_infra_perception_demo/
```

Structure:

```text
demo/style_infra_perception_demo/
  README.md
  start_demo_server.sh

  full_scene_153_233/
    assets/
    index.html
    metadata.json

  full_scene_075_152/
    assets/
    index.html
    metadata.json

  videos/
    full_scene_153_233_bev_plus_cameras.mp4
    full_scene_075_152_bev_plus_cameras.mp4
```

---

## Run demo

```bash
cd demo/style_infra_perception_demo
python -m http.server 8000
```

Open:

```text
http://localhost:8000/full_scene_153_233/
http://localhost:8000/full_scene_075_152/
```

Or:

```bash
bash scripts/pipeline/serve_demo.sh
```

---

## Demo scenes

### full_scene_153_233

Flagship scene.

Includes:

```text
3D detection boxes
online track IDs
BEV point-cloud background
camera projection
selected track playback
event episode panel
camera-side event highlights
pedestrian group crossing risk
vehicle/group future path visualization
```

### full_scene_075_152

Second clean scene.

Includes:

```text
3D detection boxes
online track IDs
BEV point-cloud background
camera projection
selected track playback
traffic event highlights
```

---

## Replay videos

```text
videos/full_scene_153_233_bev_plus_cameras.mp4
videos/full_scene_075_152_bev_plus_cameras.mp4
```

These are normal MP4 files for quick non-interactive playback.

---

## Demo order for presentation

Recommended order:

```text
1. Open full_scene_153_233.
2. Show BEV tracking and camera projection.
3. Use selected track playback.
4. Show event panel and camera-side event alert.
5. Show group-crossing risk and future path visualization.
6. Open full_scene_075_152 as second clean scene.
7. Use MP4 videos if browser interaction is not convenient.
```

---

## Validate patches

```bash
grep -R "__EVENT_EPISODE_PATCHED__\|__EVENT_CAMERA_HIGHLIGHT_PATCHED__\|__GROUP_CROSSING_PATCHED__" \
  demo/style_infra_perception_demo/full_scene_153_233/index.html \
  demo/style_infra_perception_demo/full_scene_075_152/index.html
```

Expected:

```text
full_scene_153_233: event episode + camera highlight + group crossing
full_scene_075_152: event episode + camera highlight
```
