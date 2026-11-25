"""Box utilities shared by tracking, evaluation and the gallery. Boxes are [x1, y1, x2, y2]."""
import numpy as np

ANIMALS = {"dog", "cat", "sheep", "cow", "bear", "horse"}


def iou(a, B):
    """Intersection-over-union of box a with every box in B (N x 4)."""
    B = np.asarray(B).reshape(-1, 4)
    if len(B) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def merge_animals(P, key="frame_id", iou_thr=0.6):
    """Count every COCO animal class as 'dog' and remove duplicate boxes on the same animal
    (class-agnostic non-maximum suppression: within one frame, a box overlapping a more
    confident box by IoU >= iou_thr is dropped). `key` names the frame-id column(s)."""
    P = P[P.cls.isin(ANIMALS)].sort_values("conf", ascending=False)
    keep = []
    for _, g in P.groupby(key, sort=False):
        B = g[["x1", "y1", "x2", "y2"]].to_numpy()
        alive = np.ones(len(g), bool)
        for i in range(len(g)):
            if not alive[i]:
                continue
            keep.append(g.index[i])
            alive[i + 1:] &= iou(B[i], B[i + 1:]) < iou_thr
    return P.loc[keep].assign(cls="dog")
