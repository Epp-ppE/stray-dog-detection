#!/usr/bin/env bash
# Reproduce every result in outputs/ from the raw footage, in order.
# CPU only. Times are for a 4-core CPU. On Windows, run it in Git Bash with .venv activated.
#   bash scripts/run_all.sh              # full pipeline with the final setup
#   SKIP_DOWNLOAD=1 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DET_MODEL="${DET_MODEL:-yolo11m}"; DET_SIZE="${DET_SIZE:-960}"; DET="${DET_MODEL}_${DET_SIZE}"

[ "${SKIP_DOWNLOAD:-0}" = 1 ] || bash scripts/download_data.sh        # 2.4 GB, a few minutes
mkdir -p models && (cd models && python -c "from ultralytics import YOLO; [YOLO(m + '.pt') for m in ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11s-cls']]")   # model weights (~80 MB)
cd src
python train_clock_reader.py       # ~2 min: learns the burned-in clock digits from Camera3/4
python prepare.py                  # ~10 min: duplicates, real start times -> outputs/manifest.csv
python select_eval_frames.py       # ~15 min: re-creates the 180 evaluation images (same frames: fixed seed)
python detect.py --model "$DET_MODEL" --imgsz "$DET_SIZE"   # ~2.7 h (yolo11m@960) / ~50 min (yolo11s@640)
python evaluate.py                 # ~10 min: 8 detector setups vs the 180 hand-labelled frames
python tune_tracker.py --det "$DET"                          # ~5 min: tracker settings chosen on dev clips
python track.py --det "$DET" --merge-animals                 # ~1 min
python evaluate.py                 # rescore, now including the tracked pipeline
python stats.py --tracks "${DET}_animals"                    # visits, activity, charts
python examples.py --setup "${DET}+animals"                  # success / failure gallery
python reid_demo.py --tracks "${DET}_animals" --camera Camera3_LocationC   # optional recognition demo
echo "done: see outputs/"
