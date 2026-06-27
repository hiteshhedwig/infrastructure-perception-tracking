#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
COOPDET3D_FORK_URL="${COOPDET3D_FORK_URL:-https://github.com/hiteshhedwig/coopdet3d.git}"
COOPDET3D_BRANCH="${COOPDET3D_BRANCH:-ipt-local-mods}"
if [[ ! -d .git ]]; then
  echo "ERROR: run git init first inside this repo."
  exit 1
fi
mkdir -p external
if [[ -e external/coopdet3d ]]; then
  echo "ERROR: external/coopdet3d already exists. Remove it first if needed."
  exit 1
fi
git submodule add -b "$COOPDET3D_BRANCH" "$COOPDET3D_FORK_URL" external/coopdet3d
echo "Added external/coopdet3d submodule."
