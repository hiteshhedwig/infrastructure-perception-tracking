#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


GROUP_CROSSING_JS = r'''
<script>
(function() {
  if (window.__GROUP_CROSSING_PATCHED__) return;
  window.__GROUP_CROSSING_PATCHED__ = true;

  if (!DATA.group_crossing_episodes) DATA.group_crossing_episodes = [];
  if (!DATA.group_crossing_active_episode_ids_by_frame_idx) DATA.group_crossing_active_episode_ids_by_frame_idx = {};

  const GROUP_EP_BY_ID = {};
  DATA.group_crossing_episodes.forEach(ep => {
    GROUP_EP_BY_ID[ep.episode_id] = ep;
  });

  const GROUP_BOX_EDGES = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
  ];

  function groupRiskColor(sev) {
    if (sev === "HIGH") return "#ff2222";
    if (sev === "MEDIUM") return "#ffcc00";
    return "#55ccff";
  }

  function getFrameGroupEpisodes(frame) {
    const ids = DATA.group_crossing_active_episode_ids_by_frame_idx[String(frame.frame_idx)] || [];
    return ids.map(id => GROUP_EP_BY_ID[id]).filter(Boolean);
  }

  function groupSelectedTrackId() {
    const tid = selectedTrack();
    if (tid === "ALL") return "ALL";
    return Number(tid);
  }

  function groupEpisodeMatchesSelected(ep) {
    const tid = groupSelectedTrackId();
    if (tid === "ALL") return true;

    if (Number(ep.vehicle_track_id) === tid) return true;
    return (ep.group_track_ids || []).map(Number).includes(tid);
  }

  function tracksForGroup(frame, ep) {
    const ids = new Set((ep.group_track_ids || []).map(Number));
    return frame.tracks.filter(t => ids.has(Number(t.track_id)));
  }

  function vehicleForEpisode(frame, ep) {
    return frame.tracks.find(t => Number(t.track_id) === Number(ep.vehicle_track_id));
  }

  function mean(vals) {
    if (!vals.length) return 0;
    return vals.reduce((a,b)=>a+b,0) / vals.length;
  }

  function futurePoints(x, y, vx, vy, horizon=5.0, step=1.0) {
    const pts = [];
    for (let t = 0; t <= horizon + 1e-6; t += step) {
      pts.push({t, x: x + vx * t, y: y + vy * t});
    }
    return pts;
  }

  function drawFutureDots(ctx, canvas, pts, color, label) {
    ctx.save();

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const [px, py] = worldToBev(p.x, p.y, canvas);

      ctx.beginPath();
      ctx.arc(px, py, i === 0 ? 6 : 5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.strokeStyle = "black";
      ctx.lineWidth = 3;
      ctx.stroke();

      if (i > 0) {
        const prev = pts[i-1];
        const [ppx, ppy] = worldToBev(prev.x, prev.y, canvas);

        ctx.strokeStyle = "black";
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.moveTo(ppx, ppy);
        ctx.lineTo(px, py);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 6]);
        ctx.beginPath();
        ctx.moveTo(ppx, ppy);
        ctx.lineTo(px, py);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (pts.length) {
      const last = pts[pts.length - 1];
      const [lx, ly] = worldToBev(last.x, last.y, canvas);
      ctx.font = "bold 16px Arial";
      ctx.lineWidth = 5;
      ctx.strokeStyle = "black";
      ctx.strokeText(label, lx + 8, ly - 8);
      ctx.fillStyle = color;
      ctx.fillText(label, lx + 8, ly - 8);
    }

    ctx.restore();
  }

  function closestSameTimePoint(groupPts, vehPts) {
    let best = null;

    for (let i = 0; i < Math.min(groupPts.length, vehPts.length); i++) {
      const g = groupPts[i];
      const v = vehPts[i];
      const dx = g.x - v.x;
      const dy = g.y - v.y;
      const d = Math.sqrt(dx*dx + dy*dy);

      if (!best || d < best.d) {
        best = {
          d,
          t: g.t,
          x: (g.x + v.x) / 2,
          y: (g.y + v.y) / 2
        };
      }
    }

    return best;
  }

  function drawGroupCrossingBEV(ctx, canvas, frame) {
    const episodes = getFrameGroupEpisodes(frame).filter(groupEpisodeMatchesSelected);
    if (!episodes.length) return;

    episodes.forEach(ep => {
      const color = groupRiskColor(ep.severity);
      const groupTracks = tracksForGroup(frame, ep);
      const veh = vehicleForEpisode(frame, ep);

      if (!groupTracks.length || !veh) return;

      const gx = mean(groupTracks.map(t => Number(t.x)));
      const gy = mean(groupTracks.map(t => Number(t.y)));
      const gvx = mean(groupTracks.map(t => Number(t.vx || 0)));
      const gvy = mean(groupTracks.map(t => Number(t.vy || 0)));

      const vehPts = futurePoints(Number(veh.x), Number(veh.y), Number(veh.vx || 0), Number(veh.vy || 0), 5.0, 1.0);
      const groupPts = futurePoints(gx, gy, gvx, gvy, 5.0, 1.0);
      const riskPt = closestSameTimePoint(groupPts, vehPts);

      const bevGroupPts = groupTracks.map(t => worldToBev(Number(t.x), Number(t.y), canvas));
      const xs = bevGroupPts.map(p => p[0]);
      const ys = bevGroupPts.map(p => p[1]);

      const cx = mean(xs);
      const cy = mean(ys);
      const radius = Math.max(
        24,
        ...bevGroupPts.map(p => Math.hypot(p[0] - cx, p[1] - cy)) + [24]
      ) + 16;

      ctx.save();

      // Group region.
      ctx.fillStyle = ep.severity === "HIGH" ? "rgba(255,0,0,0.14)" : "rgba(255,220,0,0.14)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = "black";
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.setLineDash([12, 7]);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);

      // Highlight individual pedestrians in group.
      groupTracks.forEach(t => {
        const [px, py] = worldToBev(Number(t.x), Number(t.y), canvas);

        ctx.beginPath();
        ctx.arc(px, py, 17, 0, 2 * Math.PI);
        ctx.strokeStyle = "black";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(px, py, 17, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.stroke();

        ctx.font = "bold 15px Arial";
        ctx.lineWidth = 4;
        ctx.strokeStyle = "black";
        ctx.strokeText(`P${t.track_id}`, px + 8, py - 8);
        ctx.fillStyle = color;
        ctx.fillText(`P${t.track_id}`, px + 8, py - 8);
      });

      // Vehicle highlight.
      const [vx, vy] = worldToBev(Number(veh.x), Number(veh.y), canvas);

      ctx.beginPath();
      ctx.arc(vx, vy, 24, 0, 2 * Math.PI);
      ctx.strokeStyle = "black";
      ctx.lineWidth = 9;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(vx, vy, 24, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.stroke();

      ctx.font = "bold 18px Arial";
      ctx.lineWidth = 5;
      ctx.strokeStyle = "black";
      ctx.strokeText(`VEH T${veh.track_id}`, vx + 10, vy - 10);
      ctx.fillStyle = color;
      ctx.fillText(`VEH T${veh.track_id}`, vx + 10, vy - 10);

      // Future vectors.
      drawFutureDots(ctx, canvas, groupPts, "#00e5ff", "group future");
      drawFutureDots(ctx, canvas, vehPts, color, "vehicle future");

      // Risk region.
      if (riskPt) {
        const [rx, ry] = worldToBev(riskPt.x, riskPt.y, canvas);

        ctx.fillStyle = ep.severity === "HIGH" ? "rgba(255,0,0,0.18)" : "rgba(255,220,0,0.18)";
        ctx.beginPath();
        ctx.arc(rx, ry, ep.severity === "HIGH" ? 32 : 28, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = "black";
        ctx.lineWidth = 9;
        ctx.beginPath();
        ctx.arc(rx, ry, ep.severity === "HIGH" ? 32 : 28, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.arc(rx, ry, ep.severity === "HIGH" ? 32 : 28, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.font = "bold 18px Arial";
        ctx.lineWidth = 5;
        ctx.strokeStyle = "black";
        ctx.strokeText(`predicted risk area`, rx + 10, ry - 10);
        ctx.fillStyle = color;
        ctx.fillText(`predicted risk area`, rx + 10, ry - 10);
      }

      // Main label.
      ctx.font = "bold 22px Arial";
      const label = `${ep.severity} CROSSING GROUP RISK | VEH T${ep.vehicle_track_id} | GROUP ${ep.group_track_ids.length} VRUs`;
      ctx.lineWidth = 7;
      ctx.strokeStyle = "black";
      ctx.strokeText(label, 22, 72);
      ctx.fillStyle = color;
      ctx.fillText(label, 22, 72);

      ctx.restore();
    });
  }

  const prevDrawBEVOverlayGroup = drawBEVOverlay;
  drawBEVOverlay = function(ctx, canvas, frame, tid) {
    prevDrawBEVOverlayGroup(ctx, canvas, frame, tid);
    drawGroupCrossingBEV(ctx, canvas, frame);
  };

  function camValidPts(track, camKey) {
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

  function camProjectedCenter(track, camKey) {
    const pts = camValidPts(track, camKey);
    if (!pts.length) return null;

    return [
      mean(pts.map(p => p[0])),
      mean(pts.map(p => p[1]))
    ];
  }

  function drawCamera3DHighlight(ctx, track, camKey, color, label) {
    if (!track || !track.camera_boxes || !track.camera_boxes[camKey]) return false;

    const proj = track.camera_boxes[camKey];
    const pts = proj.corners || [];
    const valid = proj.valid || [];
    const edges = (typeof BOX_EDGES !== "undefined") ? BOX_EDGES : GROUP_BOX_EDGES;

    let drew = false;

    edges.forEach(([a,b]) => {
      if (!valid[a] || !valid[b]) return;

      ctx.strokeStyle = "black";
      ctx.lineWidth = 10;
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

    const c = camProjectedCenter(track, camKey);
    if (c) {
      ctx.font = "bold 24px Arial";
      ctx.lineWidth = 7;
      ctx.strokeStyle = "black";
      ctx.strokeText(label, c[0] + 8, c[1] - 10);
      ctx.fillStyle = color;
      ctx.fillText(label, c[0] + 8, c[1] - 10);
    }

    return drew;
  }

  function drawGroupCrossingCameraOverlay(camKey) {
    const canvas = document.getElementById(`cam_${camKey}`);
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const frame = DATA.frames[current];

    const episodes = getFrameGroupEpisodes(frame).filter(groupEpisodeMatchesSelected);
    if (!episodes.length) return;

    episodes.forEach(ep => {
      const color = groupRiskColor(ep.severity);
      const groupTracks = tracksForGroup(frame, ep);
      const veh = vehicleForEpisode(frame, ep);

      const visibleTracks = [];
      groupTracks.forEach(t => {
        if (camValidPts(t, camKey).length) visibleTracks.push(t);
      });
      if (veh && camValidPts(veh, camKey).length) visibleTracks.push(veh);

      if (!visibleTracks.length) return;

      const allPts = [];
      visibleTracks.forEach(t => allPts.push(...camValidPts(t, camKey)));

      if (allPts.length >= 2) {
        const xs = allPts.map(p => p[0]);
        const ys = allPts.map(p => p[1]);

        const xmin = Math.max(0, Math.min(...xs) - 22);
        const ymin = Math.max(0, Math.min(...ys) - 44);
        const xmax = Math.min(canvas.width, Math.max(...xs) + 22);
        const ymax = Math.min(canvas.height, Math.max(...ys) + 22);

        const w = xmax - xmin;
        const h = ymax - ymin;

        if (w > 5 && h > 5) {
          ctx.fillStyle = ep.severity === "HIGH" ? "rgba(255,0,0,0.12)" : "rgba(255,220,0,0.12)";
          ctx.fillRect(xmin, ymin, w, h);

          ctx.strokeStyle = "black";
          ctx.lineWidth = 9;
          ctx.strokeRect(xmin, ymin, w, h);

          ctx.strokeStyle = color;
          ctx.lineWidth = 5;
          ctx.setLineDash([12, 8]);
          ctx.strokeRect(xmin, ymin, w, h);
          ctx.setLineDash([]);

          const label = `${ep.severity} CROSSING GROUP RISK`;
          ctx.font = "bold 24px Arial";
          ctx.lineWidth = 7;
          ctx.strokeStyle = "black";
          ctx.strokeText(label, xmin + 8, Math.max(28, ymin - 10));
          ctx.fillStyle = color;
          ctx.fillText(label, xmin + 8, Math.max(28, ymin - 10));
        }
      }

      groupTracks.forEach(t => {
        drawCamera3DHighlight(ctx, t, camKey, "#00e5ff", `GROUP P${t.track_id}`);
      });

      if (veh) {
        drawCamera3DHighlight(ctx, veh, camKey, color, `RISK VEH T${veh.track_id}`);
      }

      const vehC = camProjectedCenter(veh, camKey);
      const groupCenters = groupTracks.map(t => camProjectedCenter(t, camKey)).filter(Boolean);

      if (vehC && groupCenters.length) {
        const gx = mean(groupCenters.map(p => p[0]));
        const gy = mean(groupCenters.map(p => p[1]));

        ctx.strokeStyle = "black";
        ctx.lineWidth = 9;
        ctx.beginPath();
        ctx.moveTo(vehC[0], vehC[1]);
        ctx.lineTo(gx, gy);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 5;
        ctx.setLineDash([10, 7]);
        ctx.beginPath();
        ctx.moveTo(vehC[0], vehC[1]);
        ctx.lineTo(gx, gy);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  }

  const prevRenderGroupCrossing = render;
  render = function() {
    prevRenderGroupCrossing();

    // Camera drawing is async. Draw overlay several times so it lands after image load.
    ["infra_0", "infra_1", "infra_2", "vehicle_0"].forEach(cam => {
      setTimeout(() => drawGroupCrossingCameraOverlay(cam), 40);
      setTimeout(() => drawGroupCrossingCameraOverlay(cam), 140);
      setTimeout(() => drawGroupCrossingCameraOverlay(cam), 300);
    });

    updateGroupCrossingPanel();
  };

  function jumpToGroupEpisode(epId) {
    const ep = GROUP_EP_BY_ID[epId];
    if (!ep) return;

    const idx = DATA.frames.findIndex(f => Number(f.frame_idx) === Number(ep.start_frame_idx));
    if (idx < 0) return;

    current = idx;
    const slider = document.getElementById("frameSlider");
    if (slider) slider.value = current;
    render();
  }

  window.groupRiskJump = function(epId) {
    jumpToGroupEpisode(epId);
  };

  window.groupRiskNext = function() {
    for (let i = current + 1; i < DATA.frames.length; i++) {
      const eps = getFrameGroupEpisodes(DATA.frames[i]).filter(groupEpisodeMatchesSelected);
      if (eps.length) {
        current = i;
        const slider = document.getElementById("frameSlider");
        if (slider) slider.value = current;
        render();
        return;
      }
    }

    for (let i = 0; i <= current; i++) {
      const eps = getFrameGroupEpisodes(DATA.frames[i]).filter(groupEpisodeMatchesSelected);
      if (eps.length) {
        current = i;
        const slider = document.getElementById("frameSlider");
        if (slider) slider.value = current;
        render();
        return;
      }
    }
  };

  function ensureGroupPanel() {
    let panel = document.getElementById("groupCrossingPanel");
    if (panel) return panel;

    panel = document.createElement("div");
    panel.id = "groupCrossingPanel";
    panel.style.position = "fixed";
    panel.style.left = "18px";
    panel.style.bottom = "18px";
    panel.style.width = "480px";
    panel.style.maxHeight = "245px";
    panel.style.overflowY = "auto";
    panel.style.background = "rgba(0,0,0,0.78)";
    panel.style.color = "#f0f0f0";
    panel.style.border = "1px solid rgba(255,255,255,0.25)";
    panel.style.borderRadius = "10px";
    panel.style.padding = "12px";
    panel.style.fontFamily = "Arial, sans-serif";
    panel.style.fontSize = "14px";
    panel.style.zIndex = "9999";
    panel.style.boxShadow = "0 4px 20px rgba(0,0,0,0.45)";
    document.body.appendChild(panel);
    return panel;
  }

  function updateGroupCrossingPanel() {
    const panel = ensureGroupPanel();
    const frame = DATA.frames[current];
    const active = getFrameGroupEpisodes(frame).filter(groupEpisodeMatchesSelected);

    let html = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div style="font-weight:bold;font-size:16px;">Crossing Group Risk</div>
        <button onclick="groupRiskNext()" style="cursor:pointer;padding:4px 8px;border-radius:6px;border:1px solid #aaa;background:#222;color:#fff;">Next group risk</button>
      </div>
      <div style="color:#aaa;margin-bottom:8px;">
        total group risk episodes: ${DATA.group_crossing_episodes.length} | active now: ${active.length}
      </div>
    `;

    if (!active.length) {
      html += `<div style="color:#999;">No active crossing-group risk in this frame.</div>`;
    } else {
      active.forEach(ep => {
        const color = groupRiskColor(ep.severity);
        html += `
          <div style="border-left:5px solid ${color};padding:7px 8px;margin:7px 0;background:rgba(255,255,255,0.07);border-radius:6px;">
            <div style="font-weight:bold;color:${color};">${ep.severity} · CROSSING GROUP VEHICLE RISK</div>
            <div>Vehicle T${ep.vehicle_track_id} ↔ group [${ep.group_track_ids.join(", ")}]</div>
            <div>frames ${ep.start_frame_idx}-${ep.end_frame_idx} | risk distance ${Number(ep.risk_distance_m).toFixed(1)}m</div>
            <button onclick="groupRiskJump('${ep.episode_id}')" style="margin-top:5px;cursor:pointer;padding:3px 7px;border-radius:5px;border:1px solid #aaa;background:#222;color:#fff;">Jump to start</button>
          </div>
        `;
      });
    }

    panel.innerHTML = html;
  }

  updateGroupCrossingPanel();
  console.log("Group crossing risk overlay enabled.");
})();
</script>
'''


def severity_value(sev):
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return order.get(str(sev).upper(), 99)


def find_embedded_data_bounds(html):
    markers = ["const DATA =", "let DATA =", "var DATA ="]
    marker_pos = -1
    marker = None

    for m in markers:
        marker_pos = html.find(m)
        if marker_pos >= 0:
            marker = m
            break

    if marker_pos < 0:
        raise RuntimeError("Could not find embedded DATA object in index.html")

    obj_start = html.find("{", marker_pos)
    if obj_start < 0:
        raise RuntimeError("Could not find DATA object opening brace")

    depth = 0
    in_str = False
    escape = False

    for i in range(obj_start, len(html)):
        ch = html[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return obj_start, i + 1

    raise RuntimeError("Could not parse embedded DATA object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer-dir", required=True)
    ap.add_argument("--group-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-severity", default="MEDIUM", choices=["HIGH", "MEDIUM", "LOW"])
    args = ap.parse_args()

    viewer_dir = Path(args.viewer_dir)
    group_json = Path(args.group_json)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(viewer_dir, out_dir)

    meta_path = out_dir / "metadata.json"
    index_path = out_dir / "index.html"

    data = json.load(open(meta_path))
    group_payload = json.load(open(group_json))

    max_rank = severity_value(args.min_severity)

    episodes = [
        ep for ep in group_payload.get("episodes", [])
        if severity_value(ep.get("severity")) <= max_rank
    ]

    active = {}
    for ep in episodes:
        for f in range(int(ep["start_frame_idx"]), int(ep["end_frame_idx"]) + 1):
            active.setdefault(str(f), []).append(ep["episode_id"])

    data["group_crossing_episodes"] = episodes
    data["group_crossing_active_episode_ids_by_frame_idx"] = active
    data["group_crossing_source"] = str(group_json)

    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)

    html = index_path.read_text()
    obj_start, obj_end = find_embedded_data_bounds(html)
    html = html[:obj_start] + json.dumps(data, ensure_ascii=False) + html[obj_end:]

    if "__GROUP_CROSSING_PATCHED__" not in html:
        html = html.replace("</body>", GROUP_CROSSING_JS + "\n</body>")

    index_path.write_text(html)

    print("Saved group-crossing viewer:")
    print(out_dir)
    print("Injected episodes:", len(episodes))
    for ep in episodes:
        print(
            f"  {ep['episode_id']} | {ep['severity']} | "
            f"frames {ep['start_frame_idx']}-{ep['end_frame_idx']} | "
            f"vehicle T{ep['vehicle_track_id']} | group {ep['group_track_ids']}"
        )


if __name__ == "__main__":
    main()
