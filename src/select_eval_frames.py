"""Choose the hand-labelled evaluation frames — at random, BEFORE looking at detections.

Stratified by camera x day/night (infrared). Frames are spread over many clips (a cap
per clip) and spaced >= 3 s apart within a clip, because neighbouring frames are
near-duplicates. Each clip is assigned wholly to 'dev' (used to choose thresholds)
or 'test' (used only for the reported numbers), so no clip appears in both.

Images are produced by the SAME sampler as detection (video.iter_frames at
SAMPLE_FPS, 1280x720), so frame i here is pixel-identical to frame i in detect.py.
"""
import random, json
import numpy as np, pandas as pd, cv2
from config import MANIFEST, OUT, WORK, ROOT, SAMPLE_FPS, INFER_WIDTH, RAW
from video import iter_frames

SEED = 2026
# (camera, is_night): (n_frames, max per clip)
PLAN = {("Camera1_LocationA", False): (50, 2), ("Camera1_LocationA", True): (20, 2),
        ("Camera3_LocationC", False): (45, 2), ("Camera3_LocationC", True): (15, 3),
        ("Camera4_LocationD", False): (50, 6)}
MIN_GAP_S = 3.0


def clip_is_night(path):
    """Infrared clips are grayscale: mean channel spread of a middle frame < 4."""
    for i, t, f in iter_frames(path, fps=0.5, width=320, height=180):
        s = f.astype(np.int16)
        return float((s.max(2) - s.min(2)).mean()) < 4
    return False


def main():
    rng = random.Random(SEED)
    man = pd.read_csv(MANIFEST)
    man = man[man.duplicate_of.isna()].copy()
    man["night"] = [clip_is_night(ROOT / p) for p in man.path]
    picks = []
    for (cam, night), (n, cap) in PLAN.items():
        clips = man[(man.camera == cam) & (man.night == night)].video_id.tolist()
        rng.shuffle(clips)
        half = set(clips[: len(clips) // 2])                  # dev / test split by clip
        taken = {}
        attempts = 0
        while sum(len(v) for v in taken.values()) < n and attempts < 20000:
            attempts += 1
            vid = rng.choice(clips)
            if len(taken.get(vid, [])) >= cap:
                continue
            dur = float(man.loc[man.video_id == vid, "duration_s"].iloc[0])
            i = rng.randrange(0, max(1, int(dur * SAMPLE_FPS) - 1))
            if any(abs(i - j) / SAMPLE_FPS < MIN_GAP_S for j in taken.get(vid, [])):
                continue
            taken.setdefault(vid, []).append(i)
        for vid, idx in taken.items():
            for i in idx:
                picks.append(dict(frame_id=f"{vid}__{i:04d}", video_id=vid, frame=i, t=round(i / SAMPLE_FPS, 3),
                                  camera=cam, night=night, split="dev" if vid in half else "test"))
    df = pd.DataFrame(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffled labelling order
    img_dir = WORK / "eval_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for vid, g in df.groupby("video_id"):
        want = set(g.frame)
        path = ROOT / man.loc[man.video_id == vid, "path"].iloc[0]
        for i, t, f in iter_frames(path, fps=SAMPLE_FPS, width=INFER_WIDTH, height=INFER_WIDTH * 9 // 16):
            if i in want:
                cv2.imwrite(str(img_dir / f"{vid}__{i:04d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if i > max(want):
                break
    missing = [fid for fid in df.frame_id if not (img_dir / f"{fid}.jpg").exists()]
    df = df[~df.frame_id.isin(missing)]
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "eval" / "eval_frames.csv", index=False)
    print(df.groupby(["camera", "night", "split"]).size().to_string())
    print(f"{len(df)} frames from {df.video_id.nunique()} clips; missing {len(missing)}")


if __name__ == "__main__":
    main()
