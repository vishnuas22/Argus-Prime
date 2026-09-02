#!/bin/bash
# Downloads COCO val2017 and extracts to /tmp/val2017 (Argus_Master's expected path).
set -euo pipefail

ZIP=/tmp/val2017.zip
if [ ! -d /tmp/val2017 ]; then
  if [ ! -f "$ZIP" ]; then
    echo "downloading val2017 (~778MB)..."
    curl -L -o "$ZIP" http://images.cocodataset.org/zips/val2017.zip
  fi
  echo "extracting..."
  unzip -q -o "$ZIP" -d /tmp/
fi
echo "val2017 images: $(ls /tmp/val2017 | wc -l | tr -d ' ')"
