#!/usr/bin/env bash

set -euo pipefail

ROOT=${1:-data/raw/tumtraf_v2x}

echo "Dataset root: $ROOT"
echo ""

echo "Top-level folders:"
find "$ROOT" -maxdepth 1 -mindepth 1 -type d | sort
echo ""

echo "Directory structure only, depth 4:"
find "$ROOT" -maxdepth 4 -type d | sort
echo ""

echo "File extension counts:"
find "$ROOT" -type f | awk '
{
  n=split($0,a,".");
  if (n > 1) ext=tolower(a[n]);
  else ext="[no_ext]";
  count[ext]++;
}
END {
  for (e in count) print count[e], e;
}' | sort -nr
echo ""

echo "Counts by split:"
for split in train val test; do
  if [ -d "$ROOT/$split" ]; then
    echo ""
    echo "[$split]"
    echo "Images:       $(find "$ROOT/$split" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)"
    echo "Point clouds: $(find "$ROOT/$split" -type f \( -name "*.pcd" -o -name "*.bin" -o -name "*.npy" \) | wc -l)"
    echo "JSON files:   $(find "$ROOT/$split" -type f -name "*.json" | wc -l)"
    echo "YAML files:   $(find "$ROOT/$split" -type f \( -name "*.yaml" -o -name "*.yml" \) | wc -l)"
    echo "TXT files:    $(find "$ROOT/$split" -type f -name "*.txt" | wc -l)"
  fi
done

echo ""
echo "Sample files:"
echo ""
echo "Sample image:"
find "$ROOT" -type f \( -name "*.jpg" -o -name "*.png" \) | head -1
echo ""
echo "Sample point cloud:"
find "$ROOT" -type f \( -name "*.pcd" -o -name "*.bin" -o -name "*.npy" \) | head -1
echo ""
echo "Sample JSON:"
find "$ROOT" -type f -name "*.json" | head -1
