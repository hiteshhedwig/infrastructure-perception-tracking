#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


EVENT_CAMERA_JS = r'''
<script>
(function() {
  if (window.__EVENT_CAMERA_HIGHLIGHT_PATCHED__) return;
  window.__EVENT_CAMERA_HIGHLIGHT_PATCHED__ = true;

  if (!DATA.event_episodes) DATA.event_episodes = [];
  if (!DATA.active_episode_ids_by_frame_idx) DATA.active_episode_ids_by_frame_idx = {};

  const EVENT_BY_ID_CAM = {};
  DATA.event_episodes.forEach(ep => {
    EVENT_BY_ID_CAM[ep.episode_id] = ep;
  });

  const FALLBACK_BOX_EDGES = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
  ];

  function camEventColor(sev) {
    if (sev === "HIGH") return "#ff2222";
    if (sev === "MEDIUM") return "#ffcc00";
    return "#55ccff";
  }

  function camShortEventName(name) {
    if (name === "PEDESTRIAN_NEAR_VEHICLE") return "VRU NEAR VEHICLE";
    if (name === "POTENTIAL_CONFLICT") return "POTENTIAL CONFLICT";
    return name;
  }

  function camSelectedTrackId() {
    const tid = selectedTrack();
    if (tid === "ALL") return "ALL";
    return Number(tid);
  }

  function camEpisodeMatchesSelected(ep) {
    const tid = camSelectedTrackId();
    if (tid === "ALL") return true;
    return Number(ep.track_id_1) === tid || Number(ep.track_id_2) === tid;
  }

  function camActiveEpisodesForFrame(frame) {
    const ids = DATA.active_episode_ids_by_frame_idx[String(frame.frame_idx)] || [];
    return ids.map(id => EVENT_BY_ID_CAM[id]).filter(Boolean).filter(camEpisodeMatchesSelected);
  }

  function validProjectedPoints(track, camKey) {
    if (!track || !track.camera_boxes || !track.camera_boxes[camKey]) return [];

    const proj = track.camera_boxes[camKey];
    const pts = proj.corners || [];
    const valid = proj.valid || [];

    const out = [];
    for (let i = 0; i < pts.length; i++) {
      if (valid[i]) out.push([Number(pts[i][0]), Number(pts[i][1])]);
    }
    return out;
  }

  function projectedCenter(track, camKey) {
    const pts = validProjectedPoints(track, camKey);
    if (pts.length === 0) return null;

    let sx = 0, sy = 0;
    pts.forEach(p => {
      sx += p[0];
      sy += p[1];
    });

    return [sx / pts.length, sy / pts.length];
  }

  function drawEventBox3D(ctx, track, camKey, color, labelText) {
    if (!track || !track.camera_boxes || !track.camera_boxes[camKey]) return false;

    const proj = track.camera_boxes[camKey];
    const pts = proj.corners || [];
    const valid = proj.valid || [];
    const edges = (typeof BOX_EDGES !== "undefined") ? BOX_EDGES : FALLBACK_BOX_EDGES;

    let drew = false;

    edges.forEach(([a, b]) => {
      if (!valid[a] || !valid[b]) return;

      ctx.strokeStyle = "black";
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();

      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();

      drew = true;
    });

    const c = projectedCenter(track, camKey);
    if (c) {
      ctx.font = "bold 26px Arial";
      ctx.lineWidth = 7;
      ctx.strokeStyle = "black";
      ctx.strokeText(labelText, c[0] + 8, c[1] - 10);
      ctx.fillStyle = color;
      ctx.fillText(labelText, c[0] + 8, c[1] - 10);
    }

    return drew;
  }

  function drawEventRegion(ctx, canvas, ep, t1, t2, camKey) {
    const color = camEventColor(ep.severity);

    const pts = [
      ...validProjectedPoints(t1, camKey),
      ...validProjectedPoints(t2, camKey)
    ];

    if (pts.length < 2) return;

    let xs = pts.map(p => p[0]);
    let ys = pts.map(p => p[1]);

    let xmin = Math.max(0, Math.min(...xs) - 18);
    let ymin = Math.max(0, Math.min(...ys) - 34);
    let xmax = Math.min(canvas.width, Math.max(...xs) + 18);
    let ymax = Math.min(canvas.height, Math.max(...ys) + 18);

    const w = xmax - xmin;
    const h = ymax - ymin;

    if (w < 5 || h < 5) return;

    ctx.save();

    ctx.fillStyle = ep.severity === "HIGH"
      ? "rgba(255, 0, 0, 0.12)"
      : "rgba(255, 220, 0, 0.12)";
    ctx.fillRect(xmin, ymin, w, h);

    ctx.strokeStyle = "black";
    ctx.lineWidth = 8;
    ctx.strokeRect(xmin, ymin, w, h);

    ctx.strokeStyle = color;
    ctx.lineWidth = ep.severity === "HIGH" ? 5 : 4;
    ctx.setLineDash([12, 7]);
    ctx.strokeRect(xmin, ymin, w, h);
    ctx.setLineDash([]);

    const label = `${ep.severity} ${camShortEventName(ep.event_type)}`;

    const labelX = xmin + 8;
    const labelY = Math.max(28, ymin - 8);

    ctx.font = "bold 24px Arial";
    ctx.lineWidth = 7;
    ctx.strokeStyle = "black";
    ctx.strokeText(label, labelX, labelY);
    ctx.fillStyle = color;
    ctx.fillText(label, labelX, labelY);

    ctx.restore();
  }

  function drawEventPairLine(ctx, ep, t1, t2, camKey) {
    const c1 = projectedCenter(t1, camKey);
    const c2 = projectedCenter(t2, camKey);

    if (!c1 || !c2) return;

    const color = camEventColor(ep.severity);

    ctx.save();

    ctx.strokeStyle = "black";
    ctx.lineWidth = 9;
    ctx.beginPath();
    ctx.moveTo(c1[0], c1[1]);
    ctx.lineTo(c2[0], c2[1]);
    ctx.stroke();

    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.setLineDash([10, 7]);
    ctx.beginPath();
    ctx.moveTo(c1[0], c1[1]);
    ctx.lineTo(c2[0], c2[1]);
    ctx.stroke();
    ctx.setLineDash([]);

    [c1, c2].forEach(c => {
      ctx.beginPath();
      ctx.arc(c[0], c[1], ep.severity === "HIGH" ? 18 : 15, 0, 2 * Math.PI);
      ctx.strokeStyle = "black";
      ctx.lineWidth = 8;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(c[0], c[1], ep.severity === "HIGH" ? 18 : 15, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.stroke();
    });

    ctx.restore();
  }

  function drawCameraEventHighlights(ctx, canvas, frame, camKey) {
    const active = camActiveEpisodesForFrame(frame);
    if (!active.length) return;

    active.forEach(ep => {
      const t1 = frame.tracks.find(t => Number(t.track_id) === Number(ep.track_id_1));
      const t2 = frame.tracks.find(t => Number(t.track_id) === Number(ep.track_id_2));

      if (!t1 && !t2) return;

      const color = camEventColor(ep.severity);

      drawEventRegion(ctx, canvas, ep, t1, t2, camKey);
      drawEventPairLine(ctx, ep, t1, t2, camKey);

      if (t1) drawEventBox3D(ctx, t1, camKey, color, `EVENT T${t1.track_id}`);
      if (t2) drawEventBox3D(ctx, t2, camKey, color, `EVENT T${t2.track_id}`);
    });
  }

  // Replace camera drawing with event-aware camera drawing.
  drawCameraCanvas = function(camKey, serial) {
    const canvas = document.getElementById(`cam_${camKey}`);
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const frame = DATA.frames[current];
    const frameAtRequest = current;

    const imagePath = frame.camera_images[camKey];

    if (!imagePath) {
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#aaa";
      ctx.font = "22px Arial";
      ctx.fillText(`${camKey}: no image`, 30, 50);
      return;
    }

    loadImage(imagePath, (img) => {
      if (typeof renderSerial !== "undefined") {
        if (serial !== renderSerial || frameAtRequest !== current) return;
      } else {
        if (frameAtRequest !== current) return;
      }

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(0, 0, canvas.width, 32);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 16px Arial";
      ctx.fillText(camKey, 12, 22);

      if (document.getElementById("showCameraBoxes").checked) {
        const tid = selectedTrack();
        let tracks = frame.tracks;

        if (tid !== "ALL") {
          tracks = tracks.filter(t => String(t.track_id) === tid);
        }

        // Normal track boxes first.
        tracks.forEach(t => {
          const proj = t.camera_boxes[camKey];
          if (!proj) return;

          const color = tid === "ALL" ? colorForTrack(t.track_id) : "#ffff00";
          const pts = proj.corners;
          const valid = proj.valid;
          const edges = (typeof BOX_EDGES !== "undefined") ? BOX_EDGES : FALLBACK_BOX_EDGES;

          edges.forEach(([a, b]) => {
            if (!valid[a] || !valid[b]) return;

            ctx.strokeStyle = "black";
            ctx.lineWidth = 5;
            ctx.beginPath();
            ctx.moveTo(pts[a][0], pts[a][1]);
            ctx.lineTo(pts[b][0], pts[b][1]);
            ctx.stroke();

            ctx.strokeStyle = color;
            ctx.lineWidth = tid === "ALL" ? 2 : 3;
            ctx.beginPath();
            ctx.moveTo(pts[a][0], pts[a][1]);
            ctx.lineTo(pts[b][0], pts[b][1]);
            ctx.stroke();
          });

          const label = `T${t.track_id}`;
          const c = projectedCenter(t, camKey);

          if (c) {
            ctx.font = "bold 22px Arial";
            ctx.lineWidth = 6;
            ctx.strokeStyle = "black";
            ctx.strokeText(label, c[0] + 6, c[1] - 6);
            ctx.fillStyle = "white";
            ctx.fillText(label, c[0] + 6, c[1] - 6);
          }
        });

        // Event highlights last, so they stay visible.
        drawCameraEventHighlights(ctx, canvas, frame, camKey);
      }
    });
  };

  console.log("Event camera highlights enabled.");
})();
</script>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    viewer_dir = Path(args.viewer_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    shutil.copytree(viewer_dir, out_dir)

    index_path = out_dir / "index.html"
    if not index_path.exists():
        raise RuntimeError(f"index.html not found: {index_path}")

    html = index_path.read_text()

    if "__EVENT_CAMERA_HIGHLIGHT_PATCHED__" not in html:
        html = html.replace("</body>", EVENT_CAMERA_JS + "\n</body>")
    else:
        print("Camera event highlight patch already exists.")

    index_path.write_text(html)

    print("Saved camera-highlight event viewer:")
    print(out_dir)


if __name__ == "__main__":
    main()
