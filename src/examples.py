"""Step 6: success and failure gallery for the report.

Draws the hand labels and one setup's predictions (at its dev-chosen threshold) on every
evaluation frame and sorts frames into: all correct, has false alarms, has missed dogs.
Colours: green = hand-labelled dog, orange dashed = 'hard' (ignored) label,
yellow = correct detection, red = false alarm, magenta = missed dog.

usage: python examples.py --setup yolo11m_960+animals
"""
import argparse, json
import numpy as np, pandas as pd, cv2
from config import OUT, WORK
import evaluate as E

GREEN, ORANGE, YELLOW, RED, MAGENTA = (80, 200, 80), (0, 140, 255), (0, 215, 255), (40, 40, 230), (220, 60, 220)
W, H = 1280, 720


def preds_for(setup):
    if setup.endswith("+tracking"):
        return E.tracked_preds(setup[: -len("+tracking")].replace("+animals", "_animals"))
    if setup.endswith("+animals"):
        return E.merge_animals(E.predict(setup.split("+")[0]))
    P = E.predict(setup)
    return P[P.cls == "dog"]


def box(img, b, col, label=None, dashed=False):
    x1, y1, x2, y2 = (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))
    if dashed:
        for x in range(x1, x2, 14):
            cv2.line(img, (x, y1), (min(x + 7, x2), y1), col, 3); cv2.line(img, (x, y2), (min(x + 7, x2), y2), col, 3)
        for y in range(y1, y2, 14):
            cv2.line(img, (x1, y), (x1, min(y + 7, y2)), col, 3); cv2.line(img, (x2, y), (x2, min(y + 7, y2)), col, 3)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
    if label:
        (tw, th), _ = cv2.getTextSize(label, 0, 0.7, 2)
        cv2.rectangle(img, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), col, -1)
        cv2.putText(img, label, (x1 + 3, max(y1 - 5, th + 3)), 0, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def main(setup):
    m = json.loads((E.EVAL / "metrics.json").read_text())[setup]
    thr = m["threshold_from_dev"]
    labels = json.loads((E.EVAL / "labels.json").read_text())
    fr = pd.read_csv(E.EVAL / "eval_frames.csv")
    P = preds_for(setup)
    P = P[P.conf >= thr]
    out = OUT / "examples"
    for sub in ["correct", "false_alarm", "missed"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for r in fr.itertuples():
        boxes = labels[r.frame_id]["boxes"]
        G = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes if not b["hard"]]).reshape(-1, 4)
        Hd = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes if b["hard"]]).reshape(-1, 4)
        p = P[P.frame_id == r.frame_id].sort_values("conf", ascending=False)
        used = np.zeros(len(G), bool); tp, fp = [], []
        for q in p.itertuples():
            a = np.array([q.x1, q.y1, q.x2, q.y2]); o = E.iou(a, G); o[used] = -1
            if len(o) and o.max() >= 0.5:
                used[o.argmax()] = True; tp.append((a, q.conf))
            elif len(Hd) and E.iou(a, Hd).max() >= 0.5:
                continue
            else:
                fp.append((a, q.conf))
        img = cv2.imread(str(WORK / "eval_images" / f"{r.frame_id}.jpg"))
        for g in G: box(img, g, GREEN)
        for g in Hd: box(img, g, ORANGE, dashed=True)
        for a, c in tp: box(img, a, YELLOW, f"dog {c:.2f}")
        for a, c in fp: box(img, a, RED, f"false alarm {c:.2f}")
        for g in G[~used]: box(img, g, MAGENTA, "missed")
        cat = "missed" if (~used).any() else ("false_alarm" if fp else "correct")
        name = f"{r.camera[:7].lower()}_{'night' if r.night else 'day'}_{r.frame_id}.jpg"
        cv2.imwrite(str(out / cat / name), cv2.resize(img, (800, 450), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 74])
        rows.append(dict(frame_id=r.frame_id, camera=r.camera, night=r.night, split=r.split, dogs=len(G), hard=len(Hd),
                         found=int(used.sum()), missed=int((~used).sum()), false_alarms=len(fp), category=cat, image=f"{cat}/{name}"))
    df = pd.DataFrame(rows)
    df.to_csv(out / "frames_summary.csv", index=False)
    print(df.category.value_counts().to_string())
    print(df.groupby(["camera", "night"])[["dogs", "found", "missed", "false_alarms"]].sum().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="yolo11m_960+animals")
    main(ap.parse_args().setup)
