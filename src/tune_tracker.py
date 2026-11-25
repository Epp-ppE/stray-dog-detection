"""Choose ByteTrack thresholds and the track-confirmation rule on the DEV clips only.

Correct boxes from the YOLO11m@960 + animals detector often score far below the YOLO11s-era
tracker defaults (its dev-chosen box threshold is ~0.02), so the tracker thresholds are
re-tuned: a grid of settings is tracked on every dev clip and scored
by F1 of the confirmed track boxes on the dev labelled frames. The winner is written to
outputs/eval/tracker_params.json, which track.py then uses. Test frames are never looked at.

usage: python tune_tracker.py --det yolo11m_960
"""
import argparse, itertools, json
import pandas as pd
from config import OUT, WORK
from boxes import merge_animals
import track as TR
import evaluate as E


def main(det_tag):
    fr = pd.read_csv(E.EVAL / "eval_frames.csv")
    labels = json.loads((E.EVAL / "labels.json").read_text())
    dev = fr[fr.split == "dev"]
    dev_frames = [f for f in dev.frame_id if labels[f]["status"] == "done"]
    gt = {f: labels[f] for f in dev_frames}
    d = WORK / "det" / det_tag
    vids = sorted(dev.video_id.unique())
    D = pd.concat([pd.read_csv(d / f"{v}.csv") for v in vids])
    F = pd.concat([pd.read_csv(d / f"{v}.frames.csv") for v in vids])
    D = merge_animals(D, key=["video_id", "frame"])
    nf = F.groupby("video_id").frame.max() + 1
    grid = itertools.product([0.05, 0.1, 0.2, 0.3], [0.01, 0.03], [2, 3], [0.1, 0.2, 0.3, 0.45])
    rows = []
    for high, low, min_frames, min_peak in grid:
        params = dict(TR.TRACKER, track_high_thresh=high, track_low_thresh=low, new_track_thresh=high)
        T = TR.run_tracking(D, nf, params, min_frames, min_peak)
        P = E.tracked_preds(None, T=T[T.confirmed])
        s, t, n = E.match(P, gt, dev_frames)
        m = E.at_threshold(s, t, n, 0.0)
        rows.append(dict(high=high, low=low, min_frames=min_frames, min_peak=min_peak,
                         dev_precision=round(m["precision"], 3), dev_recall=round(m["recall"], 3), dev_F1=round(m["f1"], 3)))
    R = pd.DataFrame(rows).sort_values(["dev_F1", "dev_precision"], ascending=False)
    R.to_csv(E.EVAL / "tracker_grid.csv", index=False)
    b = R.iloc[0]
    best = dict(det_tag=f"{det_tag}_animals", chosen_on="dev clips, max F1 of confirmed track boxes",
                tracker=dict(track_high_thresh=float(b.high), track_low_thresh=float(b.low), new_track_thresh=float(b.high)),
                min_frames=int(b.min_frames), min_peak=float(b.min_peak),
                dev=dict(precision=float(b.dev_precision), recall=float(b.dev_recall), F1=float(b.dev_F1)))
    (E.EVAL / "tracker_params.json").write_text(json.dumps(best, indent=2))
    print(R.head(8).to_string(index=False))
    print("chosen:", json.dumps(best))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="yolo11m_960")
    main(ap.parse_args().det)
