"""Train and validate the burned-in clock reader (see clock_ocr.py).

Labels: Camera3/Camera4 filenames are Unix-ms start times, so the on-screen clock of
a frame at time t is floor(start + t + CLOCK_LEAD). Frames are sampled at phase 7 of
every 15 (t = k + 0.47 s) so labels never sit on a second boundary.
Validation: 30% of clips held out (split by clip). Writes models/clock_knn.npz and
outputs/clock_reader_eval.json.
"""
import datetime as dt, json, random, re, csv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import clock_ocr as c
from config import RAW, MODELS, OUT, TZ_HOURS

TZ = dt.timezone(dt.timedelta(hours=TZ_HOURS))


def load_clip(path):
    st = dt.datetime.fromtimestamp(int(path.stem[:13]) / 1000, TZ).replace(tzinfo=None)
    times, strips = c.read_clock_strip(path, every_n=15, phase=7)
    X = np.stack([c.cells(s) for s in strips])
    Y = np.array([c.digits_of(dt.datetime.fromtimestamp(int(st.timestamp() + t + c.CLOCK_LEAD))) for t in times])
    return path, st, X, Y


def main():
    seen, clips = set(), []
    for p in sorted(RAW.glob("Camera[34]*/*.mp4")):
        if re.fullmatch(r"\d{13}", p.stem):          # skip '(1)', '(2)' and re-shared copies
            clips.append(p)
    with ThreadPoolExecutor(4) as ex:
        D = list(ex.map(load_clip, clips))
    random.seed(1)
    idx = list(range(len(D))); random.shuffle(idx)
    test = set(idx[: len(idx) * 3 // 10])
    stack = lambda ids: (np.concatenate([D[i][2].reshape(-1, D[i][2].shape[-1]) for i in ids]),
                         np.concatenate([D[i][3].ravel() for i in ids]))
    rd = c.ClockReader(3).fit(*stack([i for i in idx if i not in test]))
    dig = []; full = []; start_err = []
    for i in test:
        path, st, X, Y = D[i]
        for x, y in zip(X, Y):
            p = rd.predict_digits(x); dig.append((p == y).mean()); full.append((p == y).all())
        s, _, _ = rd.clip_start(path)
        start_err.append(abs((s - st).total_seconds()) if s else float("inf"))
    res = dict(train_clips=len(D) - len(test), test_clips=len(test), digit_acc=float(np.mean(dig)),
               frame_timestamp_acc=float(np.mean(full)), clip_start_abs_err_median_s=float(np.median(start_err)),
               clip_start_abs_err_max_s=float(np.max(start_err)),
               clips_within_1_5s=int(sum(e <= 1.5 for e in start_err)))
    print(json.dumps(res, indent=2))
    OUT.mkdir(exist_ok=True); (OUT / "clock_reader_eval.json").write_text(json.dumps(res, indent=2))
    MODELS.mkdir(exist_ok=True)
    c.ClockReader(3).fit(*stack(idx)).save(MODELS / "clock_knn.npz")


if __name__ == "__main__":
    main()
