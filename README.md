# Stray dog detection in raw camera footage (Chiang Mai)

A complete, CPU-only pipeline that **detects and locates dogs** in unlabelled, motion-triggered
outdoor camera footage, links detections into **tracks**, and turns them into **visit and
activity statistics** per camera. It includes a hand-labelled evaluation set, a comparison
of 9 detector setups, a success and failure gallery, and a design (plus a small demo) for
**recognising individual dogs** across clips.

**Deliverables**

| What | Where |
|---|---|
| Technical report | [`report/report.pdf`](report/report.pdf) (source: [`report/report.md`](report/report.md)) |
| Presentation slides | [`slides/slides.pdf`](slides/slides.pdf), editable [`slides/slides.pptx`](slides/slides.pptx) |
| Dog recognition design | [`docs/recognition_design.md`](docs/recognition_design.md) |
| Code explained line by line | [`docs/code_walkthrough.pdf`](docs/code_walkthrough.pdf) |
| Project brief (data findings) | [`docs/project_brief.pdf`](docs/project_brief.pdf) |

---

## Results at a glance

Evaluated on **180 hand-labelled frames** (332 dogs, 28 "hard" ignore regions, 20 empty
frames) from 115 clips, chosen at random before any model output was looked at. The
threshold is tuned on the **dev** half; the numbers below are on the held-out **test** half,
with 95% bootstrap intervals.

| Setup | test AP@0.5 | precision | recall | F1 |
|---|---|---|---|---|
| YOLO11s @640, "dog" class (baseline) | 0.44 [0.36–0.52] | 0.71 | 0.44 | 0.54 |
| YOLO11m @960, "dog" class | 0.58 [0.49–0.67] | 0.84 | 0.59 | 0.69 |
| **YOLO11m @960, every animal class = dog (final)** | **0.75 [0.67–0.83]** | **0.81** | **0.75** | **0.78** |

The biggest single gain comes from counting every COCO animal class as "dog": the
pretrained detector often calls puppies "cat", and calls dogs in night-time infrared
"sheep", "cow" or "bear". At night this raises AP from 0.42 to 0.76. All 9 setups,
breakdowns by camera and day/night, and the tracking results are in
[`outputs/eval/metrics.json`](outputs/eval/metrics.json) and the report.

**Tracking and statistics.** With ByteTrack settings tuned on dev, the boxes of confirmed
tracks reach test precision **0.95** at recall 0.54 (F1 0.69), so the statistics count
almost only real dogs and are lower bounds. On all 425 clips:

| | Camera1 (feeding area) | Camera3 (water trough) | Camera4 (bench) |
|---|---|---|---|
| Clips with a dog | 98% | 92% | 100% |
| Visits (gaps ≤ 2 min merged) | 136 (45/day) | 34 (11/day) | 8 (4/day) |
| Most dogs in one frame | 4 | 7 | 4 |
| Dog-minutes on screen | 132 | 44 | 4 |
| Share of dog time moving | 11% | 16% | 23% |

Charts: [`outputs/stats/`](outputs/stats/). Recognition demo (no training): 150 Camera3
tracks grouped into 72 candidate dogs, see [`outputs/reid/`](outputs/reid/).

Two problems in the raw data were found and fixed first:
* **22 of 447 files are exact duplicates.** They are removed by file fingerprint, leaving 425 unique clips (2.44 h).
* **Camera1's filenames are export times, not recording times.** The real time is read from
  the clock burned into the video by a small digit reader trained on Camera3/4, whose
  filenames are correct. On held-out clips every start time is within 0.13 s.

---

## Repository layout

```
src/                 the pipeline (run in this order)
  config.py            paths and shared settings
  video.py             frame sampler (ffmpeg, 3 fps, 1280x720)
  clock_ocr.py         reader for the burned-in camera clock
  train_clock_reader.py   trains + validates the clock reader
  prepare.py           step 1: manifest, duplicates, real start times
  detect.py            step 2: YOLO detection on every unique clip
  select_eval_frames.py   picks the 180 evaluation frames (fixed seed)
  evaluate.py          step 5: scores 9 setups against the hand labels
  tune_tracker.py      chooses ByteTrack settings on the dev clips
  track.py             step 3: ByteTrack tracking + confirmation rule
  stats.py             step 4: visits, dogs per visit, activity, charts
  examples.py          success / failure gallery
  boxes.py             shared box utilities (IoU, merge animal classes)
  reid_demo.py         optional dog re-identification demo
scripts/
  download_data.sh     downloads the footage from the GitHub release
  run_all.sh           runs everything above in order
tools/
  label_tool.html      the browser page used for hand labelling
  import_labels.py     turns exported labels into outputs/eval/labels.json
outputs/             small results, committed (manifest, tracks, stats, metrics, gallery)
report/  slides/  docs/   the written deliverables
data/  models/       large files, NOT committed (re-created by the scripts)
```

## Setup

Python 3.10 or newer, `ffmpeg` on the PATH, about 10 GB of free disk. No GPU needed.

```bash
git clone https://github.com/Epp-ppE/stray-dog-detection.git
cd stray-dog-detection
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # small CPU build
pip install -r requirements.txt
```

`ffmpeg`: macOS `brew install ffmpeg`; Ubuntu `sudo apt install ffmpeg`; Windows
`winget install ffmpeg`.

## Reproduce everything

```bash
bash scripts/run_all.sh
```

This downloads the footage (2.4 GB) from the release `data-v1` and the model weights,
then runs every step in order. The slow step is detection: about 4.5 hours for YOLO11m
@960 on a 4-core CPU. For a quick end-to-end run, use the baseline detector (about 50 min):

```bash
DET_MODEL=yolo11s DET_SIZE=640 bash scripts/run_all.sh
```

Or run steps one by one (from `src/`):

```bash
python train_clock_reader.py && python prepare.py
python select_eval_frames.py
python detect.py --model yolo11m --imgsz 960
python evaluate.py
python tune_tracker.py --det yolo11m_960
python track.py --det yolo11m_960 --merge-animals
python stats.py --tracks yolo11m_960_animals
python examples.py --setup yolo11m_960+animals
python reid_demo.py --tracks yolo11m_960_animals --camera Camera3_LocationC
```

The hand labels (`outputs/eval/labels.json`) and the evaluation-frame list are committed,
so the evaluation reproduces exactly without relabelling. Detector predictions on the
evaluation frames are cached in `outputs/eval/preds_*.csv`; delete them to recompute.

## Method in one paragraph

Each unique clip is sampled at 3 frames per second and shrunk to 1280x720. A COCO-pretrained
YOLO11 detector (no training: there are no labels) finds animals; every animal class is
counted as "dog", and duplicate boxes on the same animal are removed. ByteTrack links boxes
into tracks within each clip, and a track counts as a dog only if it lasts long enough and
is confident at least once. The tracker settings are chosen on the dev clips. Clips with
dogs less than 2 minutes apart are merged into **visits**, giving dogs per visit,
dog-minutes by hour of day, moving versus resting time and close contact between dogs. See
the report for the reasoning behind every choice, the evaluation protocol, the failure
analysis and the limitations.

## Labelling tool

`tools/label_tool.html` is the page used to draw the evaluation boxes (it ran as a claude.ai
artifact with a small database). Labelling rules: one box per visible dog, visible extent
only, and a "hard" flag for dogs that are under a quarter visible or too unclear to judge.
`tools/import_labels.py` converts an export of its `labels` collection into
`outputs/eval/labels.json`.

## Data

Footage: GitHub release [`data-v1`](https://github.com/Epp-ppE/stray-dog-detection/releases/tag/data-v1)
(Camera1_LocationA, Camera3_LocationC, Camera4_LocationD; 6–8 January 2025). The videos
are not stored in git.
