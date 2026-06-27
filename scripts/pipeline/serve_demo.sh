#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/demo/style_infra_perception_demo"
echo "Open: http://localhost:8000/full_scene_153_233/"
echo "Open: http://localhost:8000/full_scene_075_152/"
python -m http.server 8000
