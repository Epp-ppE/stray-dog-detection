#!/usr/bin/env bash
# Download the raw footage from the GitHub release "data-v1" and unzip it into data/raw/<camera>/.
# Uses the GitHub CLI if available, otherwise plain curl on the public release URLs.
set -euo pipefail
REPO="${REPO:-Epp-ppE/stray-dog-detection}"
TAG="data-v1"
ASSETS=(Camera1_LocationA.zip Camera3_LocationC.zip Camera4_LocationD.zip)
mkdir -p data/zips data/raw
cd data/zips
for a in "${ASSETS[@]}"; do
  [ -s "$a" ] && { echo "have $a"; continue; }
  if command -v gh >/dev/null && gh release download "$TAG" -R "$REPO" -p "$a" 2>/dev/null; then :
  else curl -fL --retry 3 -o "$a" "https://github.com/$REPO/releases/download/$TAG/$a"; fi
done
cd ../..
for a in "${ASSETS[@]}"; do "${PYTHON:-python}" -m zipfile -e "data/zips/$a" data/raw; done   # Python unzip: works without the unzip tool
echo "done: $(find data/raw -type f | wc -l) files in data/raw"
