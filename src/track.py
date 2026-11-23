"""Step 3: link per-frame dog detections into tracks with ByteTrack.

Runs offline on the stored detections (detect.py), so detector thresholds and tracker
settings can be changed without re-running the CNN. Each clip is tracked on its own
(clips are separate motion-triggered recordings). A track is kept as a confirmed dog
when it lasts >= MIN_TRACK_FRAMES and its best score >= MIN_TRACK_PEAK; this removes
one-frame flickers, which are the most common false positives.

usage: python track.py --det yolo11s_640
"""
import argparse
from types import SimpleNamespace
import numpy as np, pandas as pd
from ultralytics.trackers.byte_tracker import BYTETracker
from config import WORK, OUT, SAMPLE_FPS, INFER_WIDTH

W, H = INFER_WIDTH, INFER_WIDTH * 9 // 16
TRACKER = dict(track_high_thresh=0.35, track_low_thresh=0.10, new_track_thresh=0.40,
               track_buffer=int(4 * SAMPLE_FPS),    # keep a lost dog for 4 s (occlusion by fence/plants)
               match_thresh=0.8, fuse_score=True)
MIN_TRACK_FRAMES = 3                                 # >= 1 s at 3 fps
MIN_TRACK_PEAK = 0.45
DOG_LIKE = {"dog"}                                   # see --merge-animals


class Dets:
    """Minimal Results-like container for BYTETracker (conf, cls, xywh, boolean indexing)."""
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return Dets(self.xyxy[m], self.conf[m], self.cls[m])


def track_video(dets, n_frames):
    trk = BYTETracker(SimpleNamespace(**TRACKER, tracker_type="bytetrack"))
    by_f = {f: g for f, g in dets.groupby("frame")}
    rows = []
    for f in range(n_frames):
        g = by_f.get(f)
        if g is None:
            d = Dets(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
        else:
            xyxy = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) * [W, H, W, H]
            d = Dets(xyxy, g.conf.to_numpy(np.float32), np.zeros(len(g), np.float32))
        for x1, y1, x2, y2, tid, s, _c, _i in trk.update(d):
            rows.append((f, int(tid), x1 / W, y1 / H, x2 / W, y2 / H, float(s)))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score"])


def main(det_tag, merge_animals=False):
    D = pd.read_csv(WORK / f"detections_{det_tag}.csv.gz")
    F = pd.read_csv(OUT / "frames.csv.gz")
    cls = DOG_LIKE | ({"cat", "sheep", "cow", "bear", "horse"} if merge_animals else set())
    D = D[D.cls.isin(cls)]
    nf = F.groupby("video_id").frame.max() + 1
    out = []
    for vid, n in nf.items():
        t = track_video(D[D.video_id == vid], int(n))
        if len(t):
            t.insert(0, "video_id", vid)
            out.append(t)
    T = pd.concat(out, ignore_index=True)
    T["t"] = (T.frame / SAMPLE_FPS).round(3)
    T["uid"] = T.video_id + ":" + T.track_id.astype(str)
    agg = T.groupby("uid").agg(n=("frame", "size"), peak=("score", "max"))
    good = agg[(agg.n >= MIN_TRACK_FRAMES) & (agg.peak >= MIN_TRACK_PEAK)].index
    T["confirmed"] = T.uid.isin(good)
    tag = det_tag + ("_animals" if merge_animals else "")
    T.to_csv(OUT / f"tracks_{tag}.csv.gz", index=False)
    print(f"[{tag}] {T.uid.nunique()} raw tracks, {len(good)} confirmed, "
          f"in {T[T.confirmed].video_id.nunique()} of {len(nf)} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="yolo11s_640")
    ap.add_argument("--merge-animals", action="store_true")
    a = ap.parse_args()
    main(a.det, a.merge_animals)
