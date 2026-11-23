"""Shared paths and parameters for the stray-dog pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"            # small, reproducible results (committed)
WORK = ROOT / "data" / "work"     # large intermediates (ignored)
MODELS = ROOT / "models"
MANIFEST = OUT / "manifest.csv"

TZ_HOURS = 7                       # Asia/Bangkok
SAMPLE_FPS = 3                     # frames per second analysed per clip
INFER_WIDTH = 1280                 # frames are downscaled to 1280x720 before detection
