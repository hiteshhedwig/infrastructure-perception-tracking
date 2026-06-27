import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mmcv import Config
from torchpack.utils.config import configs

COOP_ROOT = Path("external/coopdet3d").resolve()
sys.path.insert(0, str(COOP_ROOT))

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def replace_val_with_train(obj):
    if isinstance(obj, dict):
        return {k: replace_val_with_train(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_val_with_train(v) for v in obj]
    if isinstance(obj, str):
        return (
            obj.replace("tumtraf_v2x_nusc_infos_val.pkl", "tumtraf_v2x_nusc_infos_train.pkl")
               .replace("/val/", "/train/")
               .replace("/validation/", "/training/")
        )
    return obj


def make_box_corners_3d(x, y, z, l, w, h, yaw):
    """
    Use MMDet3D's own LiDARInstance3DBoxes corner convention.

    This must match CoopDet3D/visualize_coop.py. A hand-written corner
    function can be subtly wrong because of box origin/z convention.
    """
    arr = np.asarray(
        [[x, y, z, l, w, h, yaw, 0.0, 0.0]],
        dtype=np.float32,
    )
    box = LiDARInstance3DBoxes(arr, box_dim=9)
    return box.corners[0].detach().cpu().numpy()


def project_corners(corners, transform, orig_w, orig_h, out_w, out_h):
    """
    Project 3D corners using original camera calibration, then scale pixel
    coordinates to the resized image used by the HTML viewer.
    """
    corners = np.asarray(corners, dtype=np.float32)
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([corners, ones], axis=1)

    T = np.asarray(transform, dtype=np.float32)
    proj = pts_h @ T.T

    depth = proj[:, 2]
    valid = depth > 1e-3

    uv = np.zeros((corners.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = proj[valid, 0] / depth[valid]
    uv[valid, 1] = proj[valid, 1] / depth[valid]

    # Reject boxes wildly outside ORIGINAL image coordinates.
    if valid.sum() < 4:
        return None

    if (
        np.nanmax(uv[:, 0]) < -orig_w or np.nanmin(uv[:, 0]) > 2 * orig_w or
        np.nanmax(uv[:, 1]) < -orig_h or np.nanmin(uv[:, 1]) > 2 * orig_h
    ):
        return None

    # Scale from original image resolution to resized viewer resolution.
    sx = float(out_w) / float(orig_w)
    sy = float(out_h) / float(orig_h)
    uv[:, 0] *= sx
    uv[:, 1] *= sy

    return {
        "corners": [[float(a), float(b)] for a, b in uv],
        "valid": [bool(v) for v in valid],
    }


def copy_resize_image(src, dst, width=640, height=360):
    img = cv2.imread(str(src))
    if img is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        cv2.putText(canvas, "missing image", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(str(dst), canvas)
        return width, height, width, height

    orig_h, orig_w = img.shape[:2]
    resized = cv2.resize(img, (width, height))
    cv2.imwrite(str(dst), resized)
    return orig_w, orig_h, width, height


def as_jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v



def render_clean_bev_pointcloud(vehicle_points, infrastructure_points, out_path, xlim, ylim, width=900, height=900):
    """
    Render clean BEV point-cloud background only.
    No boxes, no tracks. HTML overlays tracks dynamically.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (5, 12, 5)

    # Subtle grid.
    grid_color = (30, 45, 30)
    x_min, x_max = float(xlim[0]), float(xlim[1])
    y_min, y_max = float(ylim[0]), float(ylim[1])

    def world_to_px(x, y):
        px = (x - x_min) / (x_max - x_min) * width
        py = height - (y - y_min) / (y_max - y_min) * height
        return px, py

    for gx in np.arange(np.ceil(x_min / 10) * 10, x_max + 1e-6, 10):
        px, _ = world_to_px(gx, y_min)
        cv2.line(img, (int(px), 0), (int(px), height), grid_color, 1)

    for gy in np.arange(np.ceil(y_min / 10) * 10, y_max + 1e-6, 10):
        _, py = world_to_px(x_min, gy)
        cv2.line(img, (0, int(py)), (width, int(py)), grid_color, 1)

    point_sets = []
    if vehicle_points is not None and len(vehicle_points) > 0:
        point_sets.append(np.asarray(vehicle_points))
    if infrastructure_points is not None and len(infrastructure_points) > 0:
        point_sets.append(np.asarray(infrastructure_points))

    if point_sets:
        pts = np.concatenate(point_sets, axis=0)

        # Expected dataloader format is usually [x,y,z,intensity].
        # If an unexpected batch column exists, this still gives a visual sanity check.
        if pts.shape[1] >= 3:
            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]

            mask = (
                np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
                (x >= x_min) & (x <= x_max) &
                (y >= y_min) & (y <= y_max)
            )

            x = x[mask]
            y = y[mask]
            z = z[mask]

            if len(x) > 0:
                px = ((x - x_min) / (x_max - x_min) * width).astype(np.int32)
                py = (height - (y - y_min) / (y_max - y_min) * height).astype(np.int32)

                # Height color. Clip for stable display.
                z_norm = np.clip((z - np.percentile(z, 2)) / (np.percentile(z, 98) - np.percentile(z, 2) + 1e-6), 0, 1)
                vals = (z_norm * 255).astype(np.uint8)
                colors = cv2.applyColorMap(vals.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

                # Draw as small points. Random subsample if huge.
                max_points = 120000
                if len(px) > max_points:
                    rng = np.random.default_rng(0)
                    keep = rng.choice(len(px), size=max_points, replace=False)
                    px, py, colors = px[keep], py[keep], colors[keep]

                valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                px, py, colors = px[valid], py[valid], colors[valid]

                img[py, px] = colors

                # Slightly thicken sparse points.
                img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)

    cv2.putText(
        img,
        "BEV point cloud background",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), img)



def write_index_html(out_dir, data):
    html = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Infrastructure 3D Tracking Viewer</title>
<style>
  body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
  }
  .topbar {
    padding: 10px 14px;
    background: #1c1c1c;
    display: flex;
    gap: 12px;
    align-items: center;
    border-bottom: 1px solid #333;
  }
  select, button, input {
    background: #2b2b2b;
    color: #eee;
    border: 1px solid #555;
    padding: 6px;
    border-radius: 4px;
  }
  button {
    cursor: pointer;
    min-width: 80px;
  }
  .layout {
    display: grid;
    grid-template-columns: 47vw 53vw;
    height: calc(100vh - 58px);
  }
  .left {
    padding: 10px;
    border-right: 1px solid #333;
    overflow: hidden;
  }
  .right {
    padding: 10px;
    overflow: hidden;
  }
  #bevCanvas {
    width: 100%;
    height: calc(100vh - 230px);
    background: #080c08;
    border: 1px solid #333;
  }
  .details {
    margin-top: 10px;
    background: #181818;
    border: 1px solid #333;
    padding: 10px;
    height: 130px;
    overflow: auto;
    font-size: 14px;
  }
  .camGrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 8px;
    height: calc(100vh - 90px);
  }
  .camWrap {
    position: relative;
    background: #080808;
    border: 1px solid #333;
    overflow: hidden;
  }
  .camWrap canvas {
    width: 100%;
    height: 100%;
  }
  .status {
    color: #aaa;
    font-size: 13px;
  }
</style>
</head>
<body>
<div class="topbar">
  <b>Infrastructure 3D Tracking Viewer</b>
  <span>Track:</span>
  <select id="trackSelect"></select>
  <button id="playBtn">Play</button>
  <span>Frame:</span>
  <input id="frameSlider" type="range" min="0" max="0" value="0" style="width: 330px;">
  <span id="frameLabel" class="status"></span>
  <label><input type="checkbox" id="showCameraBoxes" checked> camera boxes</label>
  <label><input type="checkbox" id="showTrails" checked> trails</label>
  <label><input type="checkbox" id="showPointCloud" checked> point cloud</label>
  <label><input type="checkbox" id="showGrid" checked> grid</label>
</div>

<div class="layout">
  <div class="left">
    <canvas id="bevCanvas" width="900" height="900"></canvas>
    <div class="details" id="details"></div>
  </div>
  <div class="right">
    <div class="camGrid">
      <div class="camWrap"><canvas id="cam_infra_0" width="640" height="360"></canvas></div>
      <div class="camWrap"><canvas id="cam_infra_1" width="640" height="360"></canvas></div>
      <div class="camWrap"><canvas id="cam_infra_2" width="640" height="360"></canvas></div>
      <div class="camWrap"><canvas id="cam_vehicle_0" width="640" height="360"></canvas></div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const BOX_EDGES = __BOX_EDGES__;

let current = 0;
let playing = false;
let timer = null;
let imgCache = {};
let imgWaiters = {};
let renderSerial = 0;

const trackSelect = document.getElementById("trackSelect");
const playBtn = document.getElementById("playBtn");
const frameSlider = document.getElementById("frameSlider");
const frameLabel = document.getElementById("frameLabel");
const details = document.getElementById("details");

function colorForTrack(id) {
  let x = Number(id) * 1103515245 + 12345;
  const r = 80 + (x & 127);
  x = (x * 1103515245 + 12345) >>> 0;
  const g = 80 + (x & 127);
  x = (x * 1103515245 + 12345) >>> 0;
  const b = 80 + (x & 127);
  return `rgb(${r},${g},${b})`;
}

function loadImage(path, cb) {
  const cached = imgCache[path];

  if (cached && cached.complete && cached.naturalWidth > 0) {
    cb(cached);
    return;
  }

  if (cached) {
    if (!imgWaiters[path]) imgWaiters[path] = [];
    imgWaiters[path].push(cb);
    return;
  }

  const img = new Image();
  imgCache[path] = img;
  imgWaiters[path] = [cb];

  img.onload = () => {
    const waiters = imgWaiters[path] || [];
    delete imgWaiters[path];
    waiters.forEach(fn => fn(img));
  };

  img.onerror = () => {
    console.warn("Failed to load image:", path);
    delete imgWaiters[path];
  };

  img.src = path;
}

function selectedTrack() {
  return trackSelect.value;
}

function visibleFramesForSelectedTrack() {
  const tid = selectedTrack();
  if (tid === "ALL") return DATA.frames.map((_, i) => i);
  const out = [];
  DATA.frames.forEach((f, i) => {
    if (f.tracks.some(t => String(t.track_id) === tid)) out.push(i);
  });
  return out;
}

function setupControls() {
  trackSelect.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "ALL";
  allOpt.textContent = "ALL TRACKS";
  trackSelect.appendChild(allOpt);

  Object.keys(DATA.tracks).sort((a,b)=>Number(a)-Number(b)).forEach(tid => {
    const t = DATA.tracks[tid];
    const opt = document.createElement("option");
    opt.value = tid;
    opt.textContent = `T${tid} | ${t.class_name} | ${t.num_frames} frames | max ${t.max_speed_mps.toFixed(1)} m/s`;
    trackSelect.appendChild(opt);
  });

  frameSlider.max = DATA.frames.length - 1;
  frameSlider.value = current;

  trackSelect.onchange = () => {
    const visible = visibleFramesForSelectedTrack();
    if (selectedTrack() !== "ALL" && visible.length) current = visible[0];
    frameSlider.value = current;
    render();
  };

  frameSlider.oninput = () => {
    current = Number(frameSlider.value);
    render();
  };

  playBtn.onclick = () => togglePlay();

  document.getElementById("showCameraBoxes").onchange = render;
  document.getElementById("showTrails").onchange = render;
  document.getElementById("showPointCloud").onchange = render;
  document.getElementById("showGrid").onchange = render;
}

function togglePlay() {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";

  if (timer) clearInterval(timer);

  if (playing) {
    timer = setInterval(() => {
      const visible = visibleFramesForSelectedTrack();
      if (!visible.length) return;

      if (selectedTrack() === "ALL") {
        current = (current + 1) % DATA.frames.length;
      } else {
        let pos = visible.indexOf(current);
        if (pos < 0) pos = 0;
        else pos = (pos + 1) % visible.length;
        current = visible[pos];
      }

      frameSlider.value = current;
      render();
    }, 100);
  }
}

function worldToBev(x, y, canvas) {
  const [xmin, xmax] = DATA.bev.xlim;
  const [ymin, ymax] = DATA.bev.ylim;
  const px = (x - xmin) / (xmax - xmin) * canvas.width;
  const py = canvas.height - (y - ymin) / (ymax - ymin) * canvas.height;
  return [px, py];
}

function bevBoxCorners(t) {
  const c = Math.cos(t.yaw);
  const s = Math.sin(t.yaw);
  const l2 = t.l / 2;
  const w2 = t.w / 2;

  const local = [
    [ l2,  w2],
    [ l2, -w2],
    [-l2, -w2],
    [-l2,  w2],
  ];

  return local.map(([lx, ly]) => {
    return [
      t.x + c * lx - s * ly,
      t.y + s * lx + c * ly,
    ];
  });
}

function drawGridLines(ctx, canvas) {
  ctx.strokeStyle = "rgba(80,110,80,0.45)";
  ctx.lineWidth = 1;
  const step = 10;

  const [xmin, xmax] = DATA.bev.xlim;
  const [ymin, ymax] = DATA.bev.ylim;

  for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) {
    const [px, _] = worldToBev(x, ymin, canvas);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, canvas.height);
    ctx.stroke();
  }

  for (let y = Math.ceil(ymin / step) * step; y <= ymax; y += step) {
    const [_, py] = worldToBev(xmin, y, canvas);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(canvas.width, py);
    ctx.stroke();
  }
}

function drawBevTitle(ctx) {
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, 0, 390, 42);
  ctx.fillStyle = "#ddd";
  ctx.font = "20px Arial";
  ctx.fillText("BEV Tracking View | x→ right, y↑ up", 18, 30);
}

function drawGrid(ctx, canvas) {
  ctx.fillStyle = "#081008";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (document.getElementById("showGrid").checked) {
    drawGridLines(ctx, canvas);
  }
  drawBevTitle(ctx);
}

function drawTrackBEV(ctx, canvas, t, strong=false) {
  const color = colorForTrack(t.track_id);
  const corners = bevBoxCorners(t).map(([x,y]) => worldToBev(x, y, canvas));

  ctx.strokeStyle = strong ? "#ffff00" : color;
  ctx.lineWidth = strong ? 5 : 3;
  ctx.beginPath();
  corners.forEach(([px, py], i) => {
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.stroke();

  const [cx, cy] = worldToBev(t.x, t.y, canvas);

  ctx.fillStyle = strong ? "#ffff00" : color;
  ctx.font = strong ? "bold 22px Arial" : "bold 18px Arial";
  ctx.fillText(`T${t.track_id} ${t.class_name} ${t.speed_mps.toFixed(1)}m/s`, cx + 6, cy - 6);

  // Future motion dots.
  const speed = Math.hypot(t.vx, t.vy);
  if (speed > 0.25) {
    [0.4, 0.8, 1.2].forEach((dt, i) => {
      const [px, py] = worldToBev(t.x + t.vx * dt, t.y + t.vy * dt, canvas);
      ctx.beginPath();
      ctx.arc(px, py, 6 - i, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function drawTrail(ctx, canvas, tid) {
  if (!document.getElementById("showTrails").checked) return;

  const pts = [];
  for (let i = 0; i <= current; i++) {
    const tr = DATA.frames[i].tracks.find(t => String(t.track_id) === String(tid));
    if (tr) pts.push(worldToBev(tr.x, tr.y, canvas));
  }

  if (pts.length < 2) return;

  ctx.strokeStyle = colorForTrack(tid);
  ctx.lineWidth = 3;
  ctx.beginPath();
  pts.forEach(([px, py], i) => {
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function drawBEVOverlay(ctx, canvas, frame, tid) {
  if (document.getElementById("showGrid").checked) {
    drawGridLines(ctx, canvas);
  }
  drawBevTitle(ctx);

  if (tid === "ALL") {
    frame.tracks.forEach(t => drawTrackBEV(ctx, canvas, t, false));
  } else {
    drawTrail(ctx, canvas, tid);
    const t = frame.tracks.find(x => String(x.track_id) === tid);
    if (t) drawTrackBEV(ctx, canvas, t, true);
  }
}

function drawBEV() {
  const canvas = document.getElementById("bevCanvas");
  const ctx = canvas.getContext("2d");
  const frame = DATA.frames[current];
  const tid = selectedTrack();

  const usePointCloud = document.getElementById("showPointCloud").checked && frame.bev_image;

  if (usePointCloud) {
    loadImage(frame.bev_image, (img) => {
      ctx.fillStyle = "#081008";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      drawBEVOverlay(ctx, canvas, frame, tid);
    });
  } else {
    drawGrid(ctx, canvas);
    if (tid === "ALL") {
      frame.tracks.forEach(t => drawTrackBEV(ctx, canvas, t, false));
    } else {
      drawTrail(ctx, canvas, tid);
      const t = frame.tracks.find(x => String(x.track_id) === tid);
      if (t) drawTrackBEV(ctx, canvas, t, true);
    }
  }
}

function drawCameraCanvas(camKey, serial) {
  const canvas = document.getElementById(`cam_${camKey}`);
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const frame = DATA.frames[current];
  const frameAtRequest = current;

  const imagePath = frame.camera_images[camKey];

  if (!imagePath) {
    ctx.fillStyle = "#111";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = "#aaa";
    ctx.font = "22px Arial";
    ctx.fillText(`${camKey}: no image`, 30, 50);
    return;
  }

  loadImage(imagePath, (img) => {
    // Critical anti-flicker guard:
    // ignore stale async image callbacks from older frames.
    if (serial !== renderSerial || frameAtRequest !== current) return;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(0,0,canvas.width,32);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 16px Arial";
    ctx.fillText(camKey, 12, 22);

    if (!document.getElementById("showCameraBoxes").checked) return;

    const tid = selectedTrack();
    let tracks = frame.tracks;
    if (tid !== "ALL") tracks = tracks.filter(t => String(t.track_id) === tid);

    tracks.forEach(t => {
      const proj = t.camera_boxes[camKey];
      if (!proj) return;

      const color = tid === "ALL" ? colorForTrack(t.track_id) : "#ffff00";
      const pts = proj.corners;
      const valid = proj.valid;

      BOX_EDGES.forEach(([a,b]) => {
        if (!valid[a] || !valid[b]) return;

        ctx.strokeStyle = "black";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = tid === "ALL" ? 3 : 4;
        ctx.beginPath();
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
        ctx.stroke();
      });

      const label = `T${t.track_id}`;
      const cx = pts.reduce((s,p)=>s+p[0],0) / pts.length;
      const cy = pts.reduce((s,p)=>s+p[1],0) / pts.length;

      ctx.font = "bold 24px Arial";
      ctx.lineWidth = 6;
      ctx.strokeStyle = "black";
      ctx.strokeText(label, cx + 6, cy - 6);
      ctx.fillStyle = "white";
      ctx.fillText(label, cx + 6, cy - 6);
    });
  });
}

function updateDetails() {
  const frame = DATA.frames[current];
  const tid = selectedTrack();

  frameLabel.textContent = `${current+1}/${DATA.frames.length} | sample ${frame.frame_idx} | ${frame.frame_id}`;

  if (tid === "ALL") {
    details.innerHTML = `
      <b>Frame:</b> ${frame.frame_idx}<br>
      <b>Frame ID:</b> ${frame.frame_id}<br>
      <b>Tracks in frame:</b> ${frame.tracks.length}<br>
      <b>Tip:</b> select a track to isolate it in BEV and camera panels.
    `;
    return;
  }

  const summary = DATA.tracks[tid];
  const t = frame.tracks.find(x => String(x.track_id) === tid);

  details.innerHTML = `
    <b>Selected:</b> T${tid}<br>
    <b>Class:</b> ${summary.class_name}<br>
    <b>Frame range:</b> ${summary.start_frame} → ${summary.end_frame}<br>
    <b>Visible frames:</b> ${summary.num_frames}<br>
    <b>Max speed:</b> ${summary.max_speed_mps.toFixed(2)} m/s<br>
    <b>Avg score:</b> ${summary.avg_score.toFixed(3)}<br>
    <hr>
    ${t ? `
      <b>Current frame:</b> ${frame.frame_idx}<br>
      <b>Current speed:</b> ${t.speed_mps.toFixed(2)} m/s<br>
      <b>Score:</b> ${t.score.toFixed(3)}<br>
      <b>Position:</b> x=${t.x.toFixed(2)}, y=${t.y.toFixed(2)}
    ` : `<b>Track not present in this frame.</b>`}
  `;
}

function render() {
  renderSerial += 1;
  const serial = renderSerial;

  drawBEV();
  drawCameraCanvas("infra_0", serial);
  drawCameraCanvas("infra_1", serial);
  drawCameraCanvas("infra_2", serial);
  drawCameraCanvas("vehicle_0", serial);
  updateDetails();
}

setupControls();
render();
</script>
</body>
</html>
'''

    html = html.replace("__DATA__", json.dumps(data))
    html = html.replace("__BOX_EDGES__", json.dumps(BOX_EDGES))
    (out_dir / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracking-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--train-as-eval", action="store_true")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    args, opts = parser.parse_known_args()

    project_root = Path.cwd()
    config_path = Path(args.config).resolve()
    tracking_csv = Path(args.tracking_csv).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    cam_dir = assets_dir / "cameras"
    bev_dir = assets_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    for cam in ["infra_0", "infra_1", "infra_2", "vehicle_0"]:
        (cam_dir / cam).mkdir(parents=True, exist_ok=True)

    os.chdir(COOP_ROOT)

    configs.load(str(config_path), recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=str(config_path))

    if args.train_as_eval:
        dataset_cfg = copy.deepcopy(cfg.data.val)
        dataset_cfg = replace_val_with_train(dataset_cfg)
    else:
        dataset_cfg = copy.deepcopy(cfg.data[args.split])

    dataset = build_dataset(dataset_cfg)
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )

    df = pd.read_csv(tracking_csv)

    if "frame_idx" in df.columns:
        frame_col = "frame_idx"
    elif "sample_idx" in df.columns:
        frame_col = "sample_idx"
    else:
        raise RuntimeError(f"Need frame_idx or sample_idx in CSV. Columns: {list(df.columns)}")

    needed = ["track_id", "class_name", "x", "y", "z", "l", "w", "h", "yaw", "vx", "vy", "speed_mps"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    selected = []
    for idx, data in enumerate(dataflow):
        if idx < args.start_index:
            continue
        if len(selected) >= args.max_frames:
            break
        selected.append((idx, data))

    print("Selected frames:", len(selected))

    frames = []

    for local_i, (sample_idx, data) in enumerate(selected):
        metas = data["metas"].data[0][0]
        rows = df[df[frame_col] == sample_idx].copy()

        # Clean point-cloud BEV background for this frame.
        bev_rel = Path("assets") / "bev" / f"{local_i:06d}.jpg"
        bev_out = out_dir / bev_rel

        try:
            vehicle_points = data["vehicle_points"].data[0][0].numpy()
            infrastructure_points = data["infrastructure_points"].data[0][0].numpy()
            pcr = cfg.point_cloud_range
            render_clean_bev_pointcloud(
                vehicle_points,
                infrastructure_points,
                bev_out,
                xlim=[float(pcr[0]), float(pcr[3])],
                ylim=[float(pcr[1]), float(pcr[4])],
                width=900,
                height=900,
            )
            bev_image = str(bev_rel)
        except Exception as e:
            print(f"WARNING: failed to render BEV pointcloud for sample {sample_idx}: {e}")
            bev_image = None

        if "time_since_update" in rows.columns:
            rows = rows[rows["time_since_update"] == 0]

        frame_id = str(rows["frame_id"].iloc[0]) if len(rows) and "frame_id" in rows.columns else str(sample_idx)

        cam_images = {}
        cam_transforms = {}
        cam_sizes = {}

        # Infra cameras.
        for k, img_path in enumerate(metas.get("infrastructure_filename", [])):
            cam_key = f"infra_{k}"
            if cam_key not in ["infra_0", "infra_1", "infra_2"]:
                continue

            src = Path(str(img_path))
            if not src.is_absolute():
                src = COOP_ROOT / src

            rel = Path("assets") / "cameras" / cam_key / f"{local_i:06d}.jpg"
            dst = out_dir / rel
            orig_w, orig_h, out_w, out_h = copy_resize_image(src, dst, args.image_width, args.image_height)

            cam_images[cam_key] = str(rel)
            cam_transforms[cam_key] = metas["infrastructure_lidar2image"][k]
            cam_sizes[cam_key] = {
                "orig_w": orig_w,
                "orig_h": orig_h,
                "out_w": out_w,
                "out_h": out_h,
            }

        # Vehicle camera.
        if "vehicle_filename" in metas:
            v2i = data["vehicle2infrastructure"].data[0].numpy().astype(np.float32)
            v2i = np.squeeze(v2i)

            for k, img_path in enumerate(metas.get("vehicle_filename", [])):
                cam_key = f"vehicle_{k}"
                if cam_key != "vehicle_0":
                    continue

                src = Path(str(img_path))
                if not src.is_absolute():
                    src = COOP_ROOT / src

                rel = Path("assets") / "cameras" / cam_key / f"{local_i:06d}.jpg"
                dst = out_dir / rel
                orig_w, orig_h, out_w, out_h = copy_resize_image(src, dst, args.image_width, args.image_height)

                cam_images[cam_key] = str(rel)
                cam_transforms[cam_key] = metas["vehicle_lidar2image"][k] @ np.linalg.inv(v2i)
                cam_sizes[cam_key] = {
                    "orig_w": orig_w,
                    "orig_h": orig_h,
                    "out_w": out_w,
                    "out_h": out_h,
                }

        track_items = []

        for _, r in rows.iterrows():
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            l, w, h = float(r["l"]), float(r["w"]), float(r["h"])
            yaw = float(r["yaw"])
            corners = make_box_corners_3d(x, y, z, l, w, h, yaw)

            cam_boxes = {}
            for cam_key, T in cam_transforms.items():
                size = cam_sizes[cam_key]
                proj = project_corners(
                    corners,
                    T,
                    size["orig_w"],
                    size["orig_h"],
                    size["out_w"],
                    size["out_h"],
                )
                if proj is not None:
                    cam_boxes[cam_key] = proj

            track_items.append({
                "track_id": int(r["track_id"]),
                "class_name": str(r["class_name"]),
                "x": x,
                "y": y,
                "z": z,
                "l": l,
                "w": w,
                "h": h,
                "yaw": yaw,
                "vx": float(r["vx"]),
                "vy": float(r["vy"]),
                "speed_mps": float(r["speed_mps"]),
                "score": float(r["last_detection_score"]) if "last_detection_score" in r and pd.notna(r["last_detection_score"]) else 0.0,
                "age": int(r["age"]) if "age" in r and pd.notna(r["age"]) else 0,
                "hits": int(r["hits"]) if "hits" in r and pd.notna(r["hits"]) else 0,
                "camera_boxes": cam_boxes,
            })

        frames.append({
            "local_idx": local_i,
            "frame_idx": int(sample_idx),
            "frame_id": frame_id,
            "bev_image": bev_image,
            "camera_images": cam_images,
            "tracks": track_items,
        })

        print(f"{local_i:03d} sample={sample_idx} tracks={len(track_items)}")

    clip_df = df[(df[frame_col] >= args.start_index) & (df[frame_col] < args.start_index + args.max_frames)].copy()

    track_summaries = {}
    for tid, g in clip_df.groupby("track_id"):
        g = g.copy()
        track_summaries[str(int(tid))] = {
            "track_id": int(tid),
            "class_name": str(g["class_name"].mode().iloc[0]) if len(g) else "OBJ",
            "start_frame": int(g[frame_col].min()),
            "end_frame": int(g[frame_col].max()),
            "num_frames": int(g[frame_col].nunique()),
            "max_speed_mps": float(g["speed_mps"].max()) if "speed_mps" in g else 0.0,
            "avg_speed_mps": float(g["speed_mps"].mean()) if "speed_mps" in g else 0.0,
            "avg_score": float(g["last_detection_score"].mean()) if "last_detection_score" in g else 0.0,
        }

    pcr = cfg.point_cloud_range
    viewer_data = {
        "project": {
            "title": "Infrastructure-Based 3D Tracking Viewer",
            "detector": "Pretrained CoopDet3D",
            "source": "tracker diagnostics + camera projections",
        },
        "bev": {
            "xlim": [float(pcr[0]), float(pcr[3])],
            "ylim": [float(pcr[1]), float(pcr[4])],
        },
        "frames": frames,
        "tracks": track_summaries,
    }

    (out_dir / "metadata.json").write_text(json.dumps(viewer_data, indent=2))
    write_index_html(out_dir, viewer_data)

    print("Done.")
    print("Viewer:", out_dir / "index.html")
    print("Metadata:", out_dir / "metadata.json")


if __name__ == "__main__":
    main()
