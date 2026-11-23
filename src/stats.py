"""Step 4: turn confirmed dog tracks into per-camera behaviour statistics.

Definitions (all from confirmed tracks, see track.py):
* dogs in a clip      = max number of confirmed tracks present in the same frame
                        (a lower bound on distinct dogs; robust to ID switches,
                        which would inflate a raw track count)
* dog-seconds         = sum over frames of dogs present / SAMPLE_FPS
* visit               = consecutive clips with dogs on one camera, merged when the gap
                        between one clip's end and the next one's start is <= VISIT_GAP_S
                        (cameras record motion-triggered bursts of 3-30 s clips)
* moving vs resting   = per track second, box-centre speed in body lengths per second
                        (box height as the body-length proxy) above/below MOVE_BL_S
* interaction         = a frame where two dogs' box centres are closer than the mean
                        of their box widths (touching / face-to-face / feeding together)
Times are local (UTC+7) recording times from manifest.start_time.

usage: python stats.py --tracks yolo11s_640
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUT, MANIFEST, SAMPLE_FPS

VISIT_GAP_S = 120
MOVE_BL_S = 0.25
CAM_SHORT = {"Camera1_LocationA": "Cam1 (A)", "Camera3_LocationC": "Cam3 (C)", "Camera4_LocationD": "Cam4 (D)"}


def main(tag):
    man = pd.read_csv(MANIFEST)
    man = man[man.duplicate_of.isna()].copy()
    man["start"] = pd.to_datetime(man.start_time)
    man["end"] = man.start + pd.to_timedelta(man.duration_s, unit="s")
    F = pd.read_csv(OUT / "frames.csv.gz")
    F["night"] = F.colourfulness < 4
    T = pd.read_csv(OUT / f"tracks_{tag}.csv.gz")
    T = T[T.confirmed]
    T["cx"], T["cy"] = (T.x1 + T.x2) / 2, (T.y1 + T.y2) / 2
    T["w"], T["h"] = T.x2 - T.x1, T.y2 - T.y1

    # ---- per clip
    per_frame = T.groupby(["video_id", "frame"]).size().rename("dogs").reset_index()
    clip = man[["video_id", "camera", "start", "end", "duration_s"]].set_index("video_id")
    clip["night"] = F.groupby("video_id").night.mean().reindex(clip.index).fillna(0) > 0.5
    clip["dogs"] = per_frame.groupby("video_id").dogs.max().reindex(clip.index).fillna(0).astype(int)
    clip["dog_seconds"] = (per_frame.groupby("video_id").dogs.sum() / SAMPLE_FPS).reindex(clip.index).fillna(0)
    clip["tracks"] = T.groupby("video_id").uid.nunique().reindex(clip.index).fillna(0).astype(int)

    # ---- activity: moving vs resting
    T = T.sort_values(["uid", "frame"])
    g = T.groupby("uid")
    step = np.hypot(g.cx.diff(), g.cy.diff() * 9 / 16) / (g.frame.diff() / SAMPLE_FPS)   # in image widths / s
    T["speed_bl"] = step / (T.h * 9 / 16).clip(lower=1e-3)                               # body lengths / s
    T["moving"] = T.speed_bl > MOVE_BL_S
    moving_frac = T.dropna(subset=["speed_bl"]).groupby("video_id").moving.mean()
    clip["moving_frac"] = moving_frac.reindex(clip.index)

    # ---- interactions
    inter = []
    for (vid, fr), gg in T.groupby(["video_id", "frame"]):
        if len(gg) < 2:
            continue
        a = gg[["cx", "cy", "w"]].to_numpy()
        hit = False
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if np.hypot(a[i, 0] - a[j, 0], (a[i, 1] - a[j, 1]) * 9 / 16) < (a[i, 2] + a[j, 2]) / 2:
                    hit = True
        inter.append((vid, fr, hit))
    I = pd.DataFrame(inter, columns=["video_id", "frame", "interaction"])
    clip["interaction_seconds"] = (I[I.interaction].groupby("video_id").size() / SAMPLE_FPS).reindex(clip.index).fillna(0)

    clip = clip.reset_index()
    clip.to_csv(OUT / "stats" / "clips.csv", index=False)

    # ---- visits
    visits = []
    for cam, c in clip[clip.dogs > 0].sort_values("start").groupby("camera"):
        cur = None
        for r in c.itertuples():
            if cur is not None and (r.start - cur["end"]).total_seconds() <= VISIT_GAP_S:
                cur["end"] = max(cur["end"], r.end); cur["clips"] += 1
                cur["max_dogs"] = max(cur["max_dogs"], r.dogs); cur["dog_seconds"] += r.dog_seconds
                cur["night"] = cur["night"] or r.night
            else:
                if cur: visits.append(cur)
                cur = dict(camera=cam, start=r.start, end=r.end, clips=1, max_dogs=r.dogs,
                           dog_seconds=r.dog_seconds, night=r.night)
        if cur: visits.append(cur)
    V = pd.DataFrame(visits)
    V["duration_s"] = (V.end - V.start).dt.total_seconds()
    V["hour"] = V.start.dt.hour
    V.to_csv(OUT / "stats" / "visits.csv", index=False)

    # ---- per camera summary
    summ = {}
    for cam, c in clip.groupby("camera"):
        v = V[V.camera == cam]
        days = (c.start.max().normalize() - c.start.min().normalize()).days + 1
        summ[cam] = dict(
            clips=int(len(c)), footage_min=round(c.duration_s.sum() / 60, 1),
            clips_with_dogs=int((c.dogs > 0).sum()), pct_clips_with_dogs=round(100 * (c.dogs > 0).mean(), 1),
            recording_span=f"{c.start.min():%Y-%m-%d %H:%M} to {c.end.max():%Y-%m-%d %H:%M}", calendar_days=int(days),
            visits=int(len(v)), visits_per_day=round(len(v) / days, 1),
            median_visit_s=round(float(v.duration_s.median()), 1) if len(v) else None,
            max_dogs_at_once=int(c.dogs.max()), mean_dogs_per_visit=round(float(v.max_dogs.mean()), 2) if len(v) else None,
            dog_minutes=round(c.dog_seconds.sum() / 60, 1),
            pct_dog_time_moving=round(100 * float(T[T.video_id.isin(c.video_id)].moving.mean()), 1),
            interaction_minutes=round(c.interaction_seconds.sum() / 60, 2),
            night_visits=int(v.night.sum()) if len(v) else 0)
    (OUT / "stats" / "summary.json").write_text(json.dumps(summ, indent=2))
    print(pd.DataFrame(summ).T.to_string())

    # ---- figures
    cams = sorted(clip.camera.unique())
    fig, axes = plt.subplots(len(cams), 1, figsize=(9, 2.3 * len(cams)), sharex=True)
    for ax, cam in zip(np.atleast_1d(axes), cams):
        c = clip[clip.camera == cam]
        rec = c.groupby(c.start.dt.hour).duration_s.sum() / 60
        dog = c.groupby(c.start.dt.hour).dog_seconds.sum() / 60
        hrs = np.arange(24)
        ax.bar(hrs, rec.reindex(hrs).fillna(0), color="#c9d3cc", label="footage recorded (min)")
        ax.bar(hrs, dog.reindex(hrs).fillna(0), color="#1f7a5a", width=0.55, label="dog-minutes detected")
        ax.set_ylabel(CAM_SHORT.get(cam, cam)); ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    np.atleast_1d(axes)[-1].set_xticks(range(0, 24, 2)); np.atleast_1d(axes)[-1].set_xlabel("hour of day (local time)")
    fig.suptitle("Dog activity by hour (all days pooled)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "stats" / "activity_by_hour.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    for k, cam in enumerate(cams):
        c = clip[clip.camera == cam]; v = V[V.camera == cam]
        ax.scatter(c.start, np.full(len(c), k), s=6, color="#c9d3cc", label="clip recorded" if k == 0 else None)
        ax.scatter(v.start, np.full(len(v), k), s=18 + 18 * v.max_dogs, color="#1f7a5a", alpha=.7,
                   label="dog visit (size = dogs)" if k == 0 else None)
    ax.set_yticks(range(len(cams)), [CAM_SHORT.get(c, c) for c in cams]); ax.set_ylim(-.6, len(cams) - .4)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Visits over the recording period", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "stats" / "visit_timeline.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="yolo11s_640")
    main(ap.parse_args().tracks)
