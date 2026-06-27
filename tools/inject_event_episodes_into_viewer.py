#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


EVENT_JS = r'''
<script>
(function() {
  if (window.__EVENT_EPISODE_PATCHED__) return;
  window.__EVENT_EPISODE_PATCHED__ = true;

  if (!DATA.event_episodes) DATA.event_episodes = [];
  if (!DATA.active_episode_ids_by_frame_idx) DATA.active_episode_ids_by_frame_idx = {};

  const EVENT_BY_ID = {};
  DATA.event_episodes.forEach(ep => {
    EVENT_BY_ID[ep.episode_id] = ep;
  });

  function eventColor(sev) {
    if (sev === "HIGH") return "#ff3333";
    if (sev === "MEDIUM") return "#ffcc00";
    return "#66ccff";
  }

  function shortEventName(name) {
    if (name === "PEDESTRIAN_NEAR_VEHICLE") return "VRU NEAR VEHICLE";
    if (name === "POTENTIAL_CONFLICT") return "POTENTIAL CONFLICT";
    return name;
  }

  function currentFrameObj() {
    return DATA.frames[current];
  }

  function activeEpisodesForFrame(frame) {
    const ids = DATA.active_episode_ids_by_frame_idx[String(frame.frame_idx)] || [];
    return ids.map(id => EVENT_BY_ID[id]).filter(Boolean);
  }

  function selectedTrackId() {
    const tid = selectedTrack();
    if (tid === "ALL") return "ALL";
    return Number(tid);
  }

  function episodeMatchesSelected(ep) {
    const tid = selectedTrackId();
    if (tid === "ALL") return true;
    return Number(ep.track_id_1) === tid || Number(ep.track_id_2) === tid;
  }

  function ensureEventPanel() {
    let panel = document.getElementById("eventEpisodePanel");
    if (panel) return panel;

    panel = document.createElement("div");
    panel.id = "eventEpisodePanel";
    panel.style.position = "fixed";
    panel.style.right = "18px";
    panel.style.bottom = "18px";
    panel.style.width = "430px";
    panel.style.maxHeight = "260px";
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

  function jumpToFrameIndex(frameIdx) {
    const i = DATA.frames.findIndex(f => Number(f.frame_idx) === Number(frameIdx));
    if (i < 0) return;

    current = i;

    const slider = document.getElementById("frameSlider");
    if (slider) slider.value = current;

    render();
  }

  window.eventv1_jumpNext = function() {
    const tid = selectedTrackId();

    for (let i = current + 1; i < DATA.frames.length; i++) {
      const eps = activeEpisodesForFrame(DATA.frames[i]).filter(episodeMatchesSelected);
      if (eps.length > 0) {
        current = i;

        const slider = document.getElementById("frameSlider");
        if (slider) slider.value = current;

        render();
        return;
      }
    }

    for (let i = 0; i <= current; i++) {
      const eps = activeEpisodesForFrame(DATA.frames[i]).filter(episodeMatchesSelected);
      if (eps.length > 0) {
        current = i;

        const slider = document.getElementById("frameSlider");
        if (slider) slider.value = current;

        render();
        return;
      }
    }
  };

  function updateEventPanel() {
    const panel = ensureEventPanel();
    const frame = currentFrameObj();
    const active = activeEpisodesForFrame(frame).filter(episodeMatchesSelected);

    const total = DATA.event_episodes.length;

    let html = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div style="font-weight:bold;font-size:16px;">Traffic Event Episodes</div>
        <button onclick="eventv1_jumpNext()" style="cursor:pointer;padding:4px 8px;border-radius:6px;border:1px solid #aaa;background:#222;color:#fff;">Next event</button>
      </div>
      <div style="color:#aaa;margin-bottom:8px;">
        total episodes: ${total} | active now: ${active.length}
      </div>
    `;

    if (active.length === 0) {
      html += `<div style="color:#999;">No active event in this frame.</div>`;
    } else {
      active.forEach(ep => {
        const color = eventColor(ep.severity);
        html += `
          <div style="border-left:5px solid ${color};padding:7px 8px;margin:7px 0;background:rgba(255,255,255,0.07);border-radius:6px;">
            <div style="font-weight:bold;color:${color};">
              ${ep.severity} · ${shortEventName(ep.event_type)}
            </div>
            <div>
              T${ep.track_id_1} ↔ T${ep.track_id_2}
              &nbsp; | &nbsp;
              frames ${ep.start_frame_idx}-${ep.end_frame_idx}
            </div>
            <div style="color:#ddd;">
              min dist: ${Number(ep.min_distance_m).toFixed(1)}m,
              min pred: ${Number(ep.min_pred_distance_m).toFixed(1)}m
            </div>
          </div>
        `;
      });
    }

    panel.innerHTML = html;
  }

  function eventv1_drawEpisodeHighlights(ctx, canvas, frame, tid) {
    const active = activeEpisodesForFrame(frame).filter(episodeMatchesSelected);

    active.forEach(ep => {
      const t1 = frame.tracks.find(t => Number(t.track_id) === Number(ep.track_id_1));
      const t2 = frame.tracks.find(t => Number(t.track_id) === Number(ep.track_id_2));
      if (!t1 || !t2) return;

      const [x1, y1] = worldToBev(t1.x, t1.y, canvas);
      const [x2, y2] = worldToBev(t2.x, t2.y, canvas);

      const color = eventColor(ep.severity);

      ctx.save();

      ctx.strokeStyle = "black";
      ctx.lineWidth = ep.severity === "HIGH" ? 10 : 8;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      ctx.strokeStyle = color;
      ctx.lineWidth = ep.severity === "HIGH" ? 5 : 4;
      ctx.setLineDash([10, 7]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.setLineDash([]);

      [ [x1, y1], [x2, y2] ].forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, ep.severity === "HIGH" ? 22 : 18, 0, 2 * Math.PI);
        ctx.strokeStyle = "black";
        ctx.lineWidth = 8;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x, y, ep.severity === "HIGH" ? 22 : 18, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.stroke();
      });

      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      const label = `${ep.severity} ${shortEventName(ep.event_type)}`;

      ctx.font = "bold 18px Arial";
      ctx.lineWidth = 5;
      ctx.strokeStyle = "black";
      ctx.strokeText(label, mx + 8, my - 8);
      ctx.fillStyle = color;
      ctx.fillText(label, mx + 8, my - 8);

      ctx.restore();
    });
  }

  const oldDrawBEVOverlay = drawBEVOverlay;
  drawBEVOverlay = function(ctx, canvas, frame, tid) {
    oldDrawBEVOverlay(ctx, canvas, frame, tid);
    eventv1_drawEpisodeHighlights(ctx, canvas, frame, tid);
  };

  const oldRender = render;
  render = function() {
    oldRender();
    updateEventPanel();
  };

  updateEventPanel();
})();
</script>
'''


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
                obj_end = i + 1
                semi = html.find(";", obj_end)
                if semi < 0:
                    semi = obj_end
                return marker_pos, obj_start, obj_end, semi + 1, marker

    raise RuntimeError("Could not parse embedded DATA object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer-dir", required=True)
    ap.add_argument("--episodes-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    viewer_dir = Path(args.viewer_dir)
    episodes_json = Path(args.episodes_json)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    shutil.copytree(viewer_dir, out_dir)

    meta_path = out_dir / "metadata.json"
    index_path = out_dir / "index.html"

    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found: {meta_path}")
    if not index_path.exists():
        raise RuntimeError(f"index.html not found: {index_path}")

    data = json.load(open(meta_path))
    ep_payload = json.load(open(episodes_json))

    data["event_episodes"] = ep_payload.get("episodes", [])
    data["active_episode_ids_by_frame_idx"] = ep_payload.get("active_episode_ids_by_frame_idx", {})
    data["event_analytics_source"] = str(episodes_json)

    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)

    html = index_path.read_text()

    marker_pos, obj_start, obj_end, stmt_end, marker = find_embedded_data_bounds(html)
    new_data_json = json.dumps(data, ensure_ascii=False)

    html = html[:obj_start] + new_data_json + html[obj_end:]

    if "__EVENT_EPISODE_PATCHED__" not in html:
        html = html.replace("</body>", EVENT_JS + "\n</body>")
    else:
        print("Event JS already present; not injecting again.")

    index_path.write_text(html)

    print("Saved event-enabled viewer:")
    print(out_dir)
    print()
    print("Episodes injected:", len(data["event_episodes"]))


if __name__ == "__main__":
    main()
