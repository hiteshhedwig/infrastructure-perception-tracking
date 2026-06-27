#!/bin/bash

set -e

echo "Creating IPT project structure..."

# Data folders
mkdir -p data/raw/tumtraf_v2x
mkdir -p data/processed/debug_subset
mkdir -p data/processed/tumtraf_v2x

# Configs
mkdir -p configs

# Source code
mkdir -p src/data
mkdir -p src/visualization
mkdir -p src/utils
mkdir -p src/tracking
mkdir -p src/detection
mkdir -p src/events
mkdir -p src/fusion

# Tools / scripts
mkdir -p tools

# Experiments / notebooks
mkdir -p notebooks

# Outputs
mkdir -p outputs/samples
mkdir -p outputs/bev
mkdir -p outputs/camera_projection
mkdir -p outputs/videos
mkdir -p outputs/metrics

# Docs and reports
mkdir -p docs
mkdir -p reports

# Docker
mkdir -p docker

# C++ modules for later
mkdir -p cpp/tracker
mkdir -p cpp/preprocessing

# Keep empty folders tracked by git
find . -type d \
  \( -path "./.git" -o -path "./data/raw/tumtraf_v2x/*" \) -prune -o \
  -type d -exec touch {}/.gitkeep \;

# Basic files
touch README.md
touch docs/dataset_notes.md
touch configs/tumtraf_v2x.yaml
touch requirements.txt

cat > .gitignore <<'GITIGNORE'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
venv/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/
data/processed/

# Outputs
outputs/
reports/*.pdf

# Models / checkpoints
*.pth
*.pt
*.onnx
*.engine
checkpoints/
runs/

# System
.DS_Store
GITIGNORE

echo "Done."
echo ""
echo "Project structure created inside: $(pwd)"
echo ""
echo "Next: place/extract the TUMTraf V2X dataset into:"
echo "  data/raw/tumtraf_v2x/"
