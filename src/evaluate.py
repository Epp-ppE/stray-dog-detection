"""Step 5: evaluate detectors against the hand labels.

Ground truth: outputs/eval/labels.json (exported from the labelling page), boxes in
normalised coords; 'hard' boxes (mostly hidden / too small to be sure) are IGNORE
regions: a detection on one is neither a true nor a false positive, and missing one
is not a false negative. Skipped frames are dropped.

Protocol
* every config predicts on the same 1280x720 frames (pixel-identical to detect.py)
* a detection is a true positive at IoU >= 0.5 with an unmatched dog box (greedy, by score)
* AP@0.5 = all-point interpolated area under the precision-recall curve (VOC/COCO style)
* the operating threshold is chosen on the DEV split (max F1) and then applied to TEST;
  all headline numbers are TEST numbers with 95% bootstrap intervals over frames
* "+animals" setups count every COCO animal class as dog (puppies are often called
  'cat'), with class-agnostic NMS so one animal is not counted twice
* the pipeline's final setup is chosen by DEV AP50, never by test numbers
* breakdowns by camera and day/night use all labelled frames (dev+test) for stability
  and are reported at each config's dev-chosen threshold

usage: python evaluate.py            (runs missing predictions, then scores)
"""
import json, argparse
import numpy as np, pandas as pd
from config import OUT, WORK, MODELS, SAMPLE_FPS
from boxes import iou, merge_animals, ANIMALS

CONFIGS = ["yolo11n_640", "yolo11s_640", "yolo11m_640", "yolo11s_960", "yolo11m_960"]
EVAL = OUT / "eval"
RNG = np.random.default_rng(0)


def predict(tag):
    from ultralytics import YOLO
    import cv2
    p = EVAL / f"preds_{tag}.csv"
    if p.exists():
        return pd.read_csv(p)
    m, sz = tag.rsplit("_", 1)
    model = YOLO(str(MODELS / f"{m}.pt"))
    names = model.names
    fr = pd.read_csv(EVAL / "eval_frames.csv")
    rows = []
    for fid in fr.frame_id:
        im = cv2.imread(str(WORK / "eval_images" / f"{fid}.jpg"))
        r = model.predict(im, imgsz=int(sz), conf=0.01, verbose=False)[0]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxyn.numpy(), r.boxes.cls.numpy().astype(int), r.boxes.conf.numpy()):
            if names[c] in ANIMALS or names[c] == "person":
                rows.append((fid, names[c], float(s), x1, y1, x2, y2))
    df = pd.DataFrame(rows, columns=["frame_id", "cls", "conf", "x1", "y1", "x2", "y2"])
    df.to_csv(p, index=False)
    return df


def tracked_preds(track_tag, T=None):
    """Boxes the full pipeline keeps after tracking (confirmed tracks only) on the eval frames."""
    T = pd.read_csv(OUT / f"tracks_{track_tag}.csv.gz") if T is None else T
    T = T[T.confirmed]
    fr = pd.read_csv(EVAL / "eval_frames.csv")
    m = T.merge(fr[["frame_id", "video_id", "frame"]], on=["video_id", "frame"])
    return pd.DataFrame(dict(frame_id=m.frame_id, cls="dog", conf=m.score, x1=m.x1, y1=m.y1, x2=m.x2, y2=m.y2))


def match(preds, gt, frames, iou_thr=0.5):
    """Return (scores, is_tp) for all non-ignored predictions, and the number of GT boxes."""
    scores, tps, n_gt = [], [], 0
    P = {k: g for k, g in preds.groupby("frame_id")}
    for fid in frames:
        boxes = gt[fid]["boxes"]
        G = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes if not b["hard"]]).reshape(-1, 4)
        Hd = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes if b["hard"]]).reshape(-1, 4)
        n_gt += len(G)
        used = np.zeros(len(G), bool)
        p = P.get(fid)
        if p is None:
            continue
        for r in p.sort_values("conf", ascending=False).itertuples():
            a = np.array([r.x1, r.y1, r.x2, r.y2])
            o = iou(a, G); o[used] = -1
            if len(o) and o.max() >= iou_thr:
                used[o.argmax()] = True; scores.append(r.conf); tps.append(True)
            elif len(Hd) and iou(a, Hd).max() >= iou_thr:
                continue                                  # on an ignore region
            else:
                scores.append(r.conf); tps.append(False)
    return np.array(scores), np.array(tps, bool), n_gt


def ap_and_curve(scores, tps, n_gt):
    o = np.argsort(-scores)
    s, t = scores[o], tps[o]
    tp, fp = np.cumsum(t), np.cumsum(~t)
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    mpre = np.concatenate([[0], prec, [0]])
    mrec = np.concatenate([[0], rec, [rec[-1] if len(rec) else 0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, s, prec, rec


def at_threshold(scores, tps, n_gt, thr):
    k = scores >= thr
    tp, fp = int(tps[k].sum()), int((~tps[k]).sum())
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / n_gt if n_gt else 1.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0, tp=tp, fp=fp, fn=fn)


def best_threshold(scores, tps, n_gt):
    cands = np.unique(np.round(scores, 3))
    best = max(cands, key=lambda t: (at_threshold(scores, tps, n_gt, t)["f1"], -t)) if len(cands) else 0.25
    return float(best)


def bootstrap(preds, gt, frames, thr, n=1000):
    frames = np.array(frames)
    aps, f1s = [], []
    for _ in range(n):
        fs = RNG.choice(frames, len(frames), replace=True)
        # duplicate frames get unique keys so each copy counts
        g2, p2 = {}, []
        for k, f in enumerate(fs):
            key = f"{f}#{k}"; g2[key] = gt[f]
            pp = preds[preds.frame_id == f]
            if len(pp): p2.append(pp.assign(frame_id=key))
        P2 = pd.concat(p2) if p2 else preds.iloc[:0]
        s, t, ng = match(P2, g2, list(g2))
        aps.append(ap_and_curve(s, t, ng)[0]); f1s.append(at_threshold(s, t, ng, thr)["f1"])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return q(aps), q(f1s)


def main(boot=1000):
    labels = json.loads((EVAL / "labels.json").read_text())
    fr = pd.read_csv(EVAL / "eval_frames.csv")
    fr = fr[fr.frame_id.map(lambda f: labels.get(f, {}).get("status") == "done")]
    gt = {f: labels[f] for f in fr.frame_id}
    dev, test = fr[fr.split == "dev"].frame_id.tolist(), fr[fr.split == "test"].frame_id.tolist()
    n_boxes = sum(len([b for b in gt[f]["boxes"] if not b["hard"]]) for f in gt)
    n_hard = sum(len([b for b in gt[f]["boxes"] if b["hard"]]) for f in gt)
    print(f"labelled frames: {len(gt)} (dev {len(dev)}, test {len(test)}); dog boxes {n_boxes}, hard {n_hard}")

    variants = {}
    for tag in CONFIGS:
        P = predict(tag)
        variants[tag] = P[P.cls == "dog"]
    for tag in ["yolo11s_640", "yolo11m_640", "yolo11m_960"]:
        variants[f"{tag}+animals"] = merge_animals(predict(tag))
    for tt in ["yolo11s_640", "yolo11m_960_animals"]:
        try:
            variants[f"{tt.replace('_animals', '+animals')}+tracking"] = tracked_preds(tt)
        except FileNotFoundError:
            pass

    res, curves = {}, {}
    for name, P in variants.items():
        sd, td, nd = match(P, gt, dev)
        thr = best_threshold(sd, td, nd)
        dev_ap = ap_and_curve(sd, td, nd)[0]
        dev_f1 = at_threshold(sd, td, nd, thr)["f1"]
        st, tt, nt = match(P, gt, test)
        ap, s, prec, rec = ap_and_curve(st, tt, nt)
        m = at_threshold(st, tt, nt, thr)
        ci_ap, ci_f1 = bootstrap(P, gt, test, thr, boot) if boot else (None, None)
        res[name] = dict(threshold_from_dev=thr, dev_AP50=round(dev_ap, 3), dev_F1=round(dev_f1, 3), test_AP50=round(ap, 3), test_AP50_CI95=ci_ap,
                         test_precision=round(m["precision"], 3), test_recall=round(m["recall"], 3),
                         test_F1=round(m["f1"], 3), test_F1_CI95=ci_f1, test_TP=m["tp"], test_FP=m["fp"], test_FN=m["fn"],
                         at_default_0_25=at_threshold(st, tt, nt, 0.25))
        curves[name] = (rec, prec)
        # breakdowns on all frames
        br = {}
        for key, sub in [("camera", "camera"), ("night", "night")]:
            for val, g in fr.groupby(sub):
                s2, t2, n2 = match(P, gt, g.frame_id.tolist())
                a2 = ap_and_curve(s2, t2, n2)[0]
                mm = at_threshold(s2, t2, n2, thr)
                br[f"{key}={val}"] = dict(frames=len(g), dogs=n2, AP50=round(a2, 3), precision=round(mm["precision"], 3),
                                          recall=round(mm["recall"], 3), FP=mm["fp"], FN=mm["fn"])
        res[name]["breakdown_all_frames"] = br
        print(f"{name:24s} dev AP50={dev_ap:.3f} F1={dev_f1:.3f} | test thr={thr:.3f}  AP50={ap:.3f} {ci_ap}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} {ci_f1}")
    (EVAL / "metrics.json").write_text(json.dumps(res, indent=2, default=float))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (r, p) in curves.items():
        ax.plot(r, p, label=f"{name} (AP50 {res[name]['test_AP50']:.2f})", lw=1.6)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Precision-recall on the test split (dog, IoU 0.5)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(EVAL / "pr_curves.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--predict-only", action="store_true")
    a = ap.parse_args()
    if a.predict_only:
        for t in CONFIGS:
            predict(t); print("predicted", t)
    else:
        main(a.boot)
