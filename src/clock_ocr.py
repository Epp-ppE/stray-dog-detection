"""Read the burned-in camera clock (e.g. '2025/01/07 06:22:29', top-left of frame).

Why not plain Tesseract: the overlay is white text with a thin dark outline. On
bright daytime backgrounds the fill blends into the sky and Tesseract fails
(2/20 clips read in a test). The font, size and position are fixed, so instead each
of the 14 digit cells is cut out at a known position and classified by a small
k-NN on HOG features.

Training labels come for free: Camera3/Camera4 filenames are Unix-ms start times,
so the clock value of every frame in those clips is known (start + t + offset).
The trained reader is checked on held-out clips and then used for Camera1, whose
filenames are export times, not recording times.
"""
import datetime as dt
import subprocess
import numpy as np
import cv2
from skimage.feature import hog

FULL_W, FULL_H = 2560, 1440
CROP = (120, 40, 940, 120)            # x0, y0, x1, y1 in full-res pixels
# on-screen clock = floor(filename_start + t + CLOCK_LEAD) (measured on Cam3 night clips)
CLOCK_LEAD = 0.9
PITCH, X0, CELL_W, Y0, Y1 = 42, 128, 40, 48, 110
DIGIT_POS = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]   # YYYY MM DD HH MM SS


def read_clock_strip(path, every_n=15, fps=15, phase=0):
    """Decode only the clock strip of every n-th frame. Returns (times_s, gray_strips)."""
    x0, y0, x1, y1 = CROP
    w, h = x1 - x0, y1 - y0
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-an",
           "-vf", f"select='eq(mod(n\\,{every_n})\\,{phase})',crop={w}:{h}:{x0}:{y0},format=gray",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    strips = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    times = (np.arange(len(strips)) * every_n + phase) / fps
    return times, strips


def cells(strip):
    """Cut the 14 digit cells out of one clock strip -> (14, feat) HOG features."""
    out = []
    for k in DIGIT_POS:
        xa = X0 + PITCH * k - CROP[0]
        c = strip[Y0 - CROP[1]:Y1 - CROP[1], xa:xa + CELL_W]
        c = cv2.resize(c, (32, 48), interpolation=cv2.INTER_AREA)
        c = cv2.equalizeHist(c)
        out.append(hog(c, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.stack(out)


def digits_of(t: dt.datetime):
    return [int(ch) for ch in t.strftime("%Y%m%d%H%M%S")]


class ClockReader:
    def __init__(self, k=3):
        self.k = k
        self.X = self.y = None

    def fit(self, X, y):
        self.X = X.astype(np.float32)
        self.X /= np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-6
        self.y = np.asarray(y)
        return self

    def predict_digits(self, F):
        F = F.astype(np.float32)
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6
        sim = F @ self.X.T
        idx = np.argsort(-sim, axis=1)[:, :self.k]
        votes = self.y[idx]
        return np.array([np.bincount(v, minlength=10).argmax() for v in votes])

    def read(self, strip):
        d = "".join(map(str, self.predict_digits(cells(strip))))
        try:
            return dt.datetime.strptime(d, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def clip_start(self, path):
        """Robust clip start: median over all sampled frames of (reading - t)."""
        times, strips = read_clock_strip(path, phase=7)
        starts = []
        for t, s in zip(times, strips):
            r = self.read(s)
            if r is not None:
                starts.append(r.timestamp() - t)
        if not starts:
            return None, 0, len(times)
        # clock shows floor(true + lead); +0.5 recentres the 1 s quantisation, -lead removes the lead
        med = float(np.median(starts)) + 0.5 - CLOCK_LEAD
        agree = sum(abs(s - med) <= 2 for s in starts)
        return dt.datetime.fromtimestamp(med), agree, len(times)

    def save(self, p):
        np.savez_compressed(p, X=self.X, y=self.y, k=self.k)

    @classmethod
    def load(cls, p):
        z = np.load(p)
        r = cls(int(z["k"]))
        r.X, r.y = z["X"], z["y"]
        return r
