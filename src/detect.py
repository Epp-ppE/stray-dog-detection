"""Step 2: run a pretrained COCO detector on sampled frames of every unique clip.

Detections are stored with a LOW confidence floor (0.01) so that thresholds can be
tuned later without re-running inference. Besides 'dog', other animal classes and
'person' are kept to study confusions (e.g. a dog predicted as 'cat' or 'sheep').
Boxes are normalised to [0, 1]. Per-frame colourfulness is stored to separate
infrared (grayscale) night frames from day frames.

usage: python detect.py --model yolo11s --imgsz 640
"""
import argparse, time
import numpy as np, pandas as pd
from ultralytics import YOLO
from config import MANIFEST, MODELS, WORK, OUT, SAMPLE_FPS, INFER_WIDTH, ROOT
from video import iter_frames

KEEP = {0: "person", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 21: "bear"}
CONF_FLOOR = 0.01   # the best setup's dev-chosen threshold is ~0.02, so keep boxes down to 0.01
BATCH = 8


def colourfulness(f):
    s = f[::8, ::8].astype(np.int16)
    return float((s.max(2) - s.min(2)).mean())


def run(model_name, imgsz):
    tag = f"{model_name}_{imgsz}"
    out_dir = WORK / "det" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODELS / f"{model_name}.pt"))
    man = pd.read_csv(MANIFEST)
    man = man[man.duplicate_of.isna()]
    # clips that contain hand-labelled frames first, so tracker tuning can start early
    ev = OUT / "eval" / "eval_frames.csv"
    if ev.exists():
        first = set(pd.read_csv(ev).video_id)
        man = man.assign(_p=~man.video_id.isin(first)).sort_values(["_p", "video_id"]).drop(columns="_p")
    t0 = time.time()
    for k, r in enumerate(man.itertuples()):
        dpath, fpath = out_dir / f"{r.video_id}.csv", out_dir / f"{r.video_id}.frames.csv"
        if dpath.exists() and fpath.exists():
            continue
        dets, frames, buf = [], [], []

        def flush():
            res = model.predict([b[2] for b in buf], imgsz=imgsz, conf=CONF_FLOOR,
                                classes=list(KEEP), verbose=False)
            for (i, t, f), rr in zip(buf, res):
                b = rr.boxes
                xyxyn = b.xyxyn.numpy()
                for (x1, y1, x2, y2), c, s in zip(xyxyn, b.cls.numpy().astype(int), b.conf.numpy()):
                    dets.append((r.video_id, i, round(t, 3), KEEP[c], round(float(s), 4),
                                 round(float(x1), 5), round(float(y1), 5), round(float(x2), 5), round(float(y2), 5)))
            buf.clear()

        for i, t, f in iter_frames(ROOT / r.path, fps=SAMPLE_FPS, width=INFER_WIDTH, height=INFER_WIDTH * 9 // 16):
            frames.append((r.video_id, i, round(t, 3), round(colourfulness(f), 2)))
            buf.append((i, t, f))
            if len(buf) == BATCH:
                flush()
        if buf:
            flush()
        pd.DataFrame(dets, columns=["video_id", "frame", "t", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(dpath, index=False)
        pd.DataFrame(frames, columns=["video_id", "frame", "t", "colourfulness"]).to_csv(fpath, index=False)
        if k % 20 == 0:
            el = time.time() - t0
            print(f"[{tag}] {k + 1}/{len(man)} clips, {el / 60:.1f} min", flush=True)
    # gather
    D = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*[!s].csv")) if not p.name.endswith(".frames.csv")])
    F = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*.frames.csv"))])
    D.to_csv(WORK / f"detections_{tag}.csv.gz", index=False)
    F.to_csv(OUT / "frames.csv.gz", index=False)
    print(f"[{tag}] done: {len(F)} frames, {len(D)} raw detections, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    run(a.model, a.imgsz)
