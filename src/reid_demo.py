"""Dog re-identification demo (optional part of the assignment; design in docs/recognition_design.md).

For each confirmed daytime track: take its best crops (largest, most confident boxes, not
touching the image edge, not overlapping another dog), embed them with a pretrained ImageNet
classifier (YOLO11s-cls, 512-d pooled features) plus a coat-colour histogram, average them
per track, centre each feature type on the camera mean and L2-normalise. Tracks are then
grouped with average-linkage agglomerative clustering on cosine similarity. Tracks visible in
the same frame are never merged ("cannot-link": they must be different dogs).

No identity labels exist, so the output is qualitative: image grids of the multi-clip
groups and nearest-neighbour examples, for a person to check. This is the "baseline (no
training)" stage of the design; it is NOT a validated identity system.

usage: python reid_demo.py --tracks yolo11m_960_animals --camera Camera3_LocationC
"""
import argparse, json, pickle
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
from config import OUT, MODELS, ROOT, MANIFEST, SAMPLE_FPS, INFER_WIDTH, WORK
from boxes import iou
from video import iter_frames

W, H = INFER_WIDTH, INFER_WIDTH * 9 // 16
CROPS_PER_TRACK = 4
MIN_CROP_H = 0.12          # crop must be at least 12% of the frame height (~86 px)
LINK_DIST = 0.5            # merge two groups when their mean cosine similarity exceeds 1 - LINK_DIST


def best_crops(T):
    """Pick up to CROPS_PER_TRACK frames per track: big, confident, away from the image edge."""
    T = T.assign(h=T.y2 - T.y1, area=(T.x2 - T.x1) * (T.y2 - T.y1))
    T = T[(T.h >= MIN_CROP_H) & (T.x1 > 0.01) & (T.x2 < 0.99) & (T.y2 < 0.995)]
    # drop crops that overlap another dog in the same frame: they would mix two dogs' features
    alone = []
    for _, g in T.groupby(["video_id", "frame"]):
        B = g[["x1", "y1", "x2", "y2"]].to_numpy()
        for k in range(len(g)):
            o = iou(B[k], np.delete(B, k, 0))
            alone.append(not (len(o) and o.max() > 0.05))
    T = T[np.array(alone)] if len(alone) else T
    T = T.assign(q=T.area * T.score)
    return T.sort_values("q", ascending=False).groupby("uid").head(CROPS_PER_TRACK)


def extract(sel, man):
    crops = {}
    for vid, g in sel.groupby("video_id"):
        want = {int(f): rows for f, rows in g.groupby("frame")}
        for i, t, f in iter_frames(ROOT / man.loc[vid, "path"], fps=SAMPLE_FPS, width=W, height=H):
            for r in want.get(i, pd.DataFrame()).itertuples():
                px, py = 0.08 * (r.x2 - r.x1), 0.08 * (r.y2 - r.y1)
                x1, y1 = int(max(0, r.x1 - px) * W), int(max(0, r.y1 - py) * H)
                x2, y2 = int(min(1, r.x2 + px) * W), int(min(1, r.y2 + py) * H)
                crops.setdefault(r.uid, []).append(f[y1:y2, x1:x2].copy())
            if i > max(want):
                break
    return crops


def cluster(E, cannot):
    """Average-linkage agglomerative clustering on cosine distance with cannot-link pairs."""
    groups = [[i] for i in range(len(E))]
    S = E @ E.T
    while True:
        best, pair = 1 - LINK_DIST, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if any((i, j) in cannot for i in groups[a] for j in groups[b]):
                    continue
                s = S[np.ix_(groups[a], groups[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if pair is None:
            return groups
        a, b = pair
        groups[a] += groups[b]
        del groups[b]


def colour_hist(c):
    """Coat colour: hue-saturation histogram of the crop (12 x 6 bins), square-rooted."""
    h = cv2.cvtColor(cv2.resize(c, (64, 96)), cv2.COLOR_BGR2HSV)
    x = cv2.calcHist([h], [0, 1], None, [12, 6], [0, 180, 0, 256]).ravel()
    return np.sqrt(x / (x.sum() + 1e-6))


def features(E, crops, uids):
    """Centre each feature type on this camera's mean (generic 'dog-ness' is shared by every
    track and would otherwise dominate the similarity), then join CNN + colour features."""
    Hc = np.stack([np.mean([colour_hist(c) for c in crops[u]], 0) for u in uids])
    out = []
    for M in (E, Hc):
        M = M - M.mean(0)
        out.append(M / np.linalg.norm(M, axis=1, keepdims=True))
    Z = np.hstack(out)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def tile(imgs, size=150):
    out = []
    for im in imgs:
        h, w = im.shape[:2]; s = size / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.full((size, size, 3), 245, np.uint8)
        y, x = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        out.append(canvas)
    return np.hstack(out)


def main(tracks, camera, link_dist=None):
    global LINK_DIST
    LINK_DIST = link_dist or LINK_DIST
    man = pd.read_csv(MANIFEST).set_index("video_id")
    man["start"] = pd.to_datetime(man.start_time)
    F = pd.read_csv(OUT / "frames.csv.gz")
    night = F.groupby("video_id").colourfulness.mean() < 4
    T = pd.read_csv(OUT / f"tracks_{tracks}.csv.gz")
    T = T[T.confirmed & T.video_id.isin(man.index[man.camera == camera]) & ~T.video_id.map(night).fillna(False)]
    cache = WORK / f"reid_cache_{tracks}_{camera}.pkl"
    if cache.exists():
        crops, uids, E = pickle.loads(cache.read_bytes())
    else:
        sel = best_crops(T)
        sel = sel[sel.uid.map(sel.uid.value_counts()) >= 2]            # need at least 2 good crops
        crops = extract(sel, man)
        model = YOLO(str(MODELS / "yolo11s-cls.pt"))
        uids = sorted(crops)
        E = []
        for u in uids:
            v = np.stack([model.embed(c, verbose=False)[0].numpy() for c in crops[u]]).mean(0)
            E.append(v / np.linalg.norm(v))
        E = np.stack(E)
        cache.write_bytes(pickle.dumps((crops, uids, E)))
    E = features(E, crops, uids)
    # cannot-link: tracks sharing any frame of the same clip are different dogs
    occ = T[T.uid.isin(uids)].groupby(["video_id", "frame"]).uid.apply(list)
    idx = {u: i for i, u in enumerate(uids)}
    cannot = set()
    for us in occ:
        for a in us:
            for b in us:
                if a != b:
                    cannot.add((idx[a], idx[b]))
    groups = sorted(cluster(E, cannot), key=len, reverse=True)

    out = OUT / "reid"
    out.mkdir(exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    rows, shown = [], 0
    for gi, g in enumerate(groups):
        us = [uids[i] for i in g]
        vids = {u.split(":")[0] for u in us}
        for u in us:
            vid = u.split(":")[0]
            rows.append(dict(group=gi, uid=u, video_id=vid, start=str(man.loc[vid, "start"]), n_clips_in_group=len(vids)))
        if len(vids) >= 2 and shown < 8:
            strips = []
            for u in sorted(us, key=lambda u: man.loc[u.split(':')[0], "start"])[:6]:
                st = tile(crops[u][:CROPS_PER_TRACK])
                lab = f"{man.loc[u.split(':')[0], 'start']:%d %b %H:%M:%S}  track {u.split(':')[1]}"
                bar = np.full((26, st.shape[1], 3), 255, np.uint8)
                cv2.putText(bar, lab, (6, 19), 0, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
                strips.append(np.vstack([bar, st]))
            wmax = max(s.shape[1] for s in strips)
            strips = [np.hstack([s, np.full((s.shape[0], wmax - s.shape[1], 3), 255, np.uint8)]) for s in strips]
            cv2.imwrite(str(out / f"group_{shown + 1:02d}.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 82])
            shown += 1
    pd.DataFrame(rows).to_csv(out / "groups.csv", index=False)
    multi = [g for g in groups if len({uids[i].split(':')[0] for i in g}) >= 2]
    summary = dict(camera=camera, tracks_embedded=len(uids), groups=len(groups), groups_spanning_2plus_clips=len(multi),
                   link_distance=LINK_DIST, model="yolo11s-cls (ImageNet) pooled features, no fine-tuning",
                   note="No identity labels: groups are candidates for a person to verify, not confirmed identities.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="yolo11m_960_animals")
    ap.add_argument("--camera", default="Camera3_LocationC")
    ap.add_argument("--link-dist", type=float, default=None)
    a = ap.parse_args()
    main(a.tracks, a.camera, a.link_dist)
