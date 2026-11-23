"""Step 1: build outputs/manifest.csv — one row per video file.

* probes each file (duration, fps, resolution, codecs)
* finds exact duplicates by MD5 (Camera4 '(1)'/'(2)' copies, a re-shared Camera3 file)
* assigns the real recording start time:
    - Camera3/Camera4: filename is Unix-ms start time (validated against the clock)
    - Camera1: filenames are export times, so the burned-in clock is read (clock_ocr)
"""
import csv, datetime as dt, hashlib, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from config import RAW, MANIFEST, MODELS, TZ_HOURS
from clock_ocr import ClockReader

TZ = dt.timezone(dt.timedelta(hours=TZ_HOURS))


def probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                                   "-show_streams", str(path)], capture_output=True, text=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = map(int, v["avg_frame_rate"].split("/"))
    return dict(duration_s=round(float(j["format"]["duration"]), 3), fps=round(num / den, 3),
                width=v["width"], height=v["height"], codec=v["codec_name"])


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = sorted(p for p in RAW.glob("*/*") if p.suffix.lower() == ".mp4")
    print(f"{len(files)} files")
    with ThreadPoolExecutor(4) as ex:
        info = list(ex.map(lambda p: {**probe(p), "md5": md5(p)}, files))
    rows, first = [], {}
    for p, inf in zip(files, info):
        cam = p.parent.name
        vid = f"{cam.split('_')[0].replace('Camera', 'cam')}_{p.stem}"
        dup = first.setdefault(inf["md5"], vid)
        rows.append(dict(video_id=vid, camera=cam, path=str(p.relative_to(RAW.parents[1])), **inf,
                         duplicate_of="" if dup == vid else dup))
    # prefer the cleanly named file as the canonical copy of a duplicate group
    groups = {}
    for r in rows:
        groups.setdefault(r["md5"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: (not re.fullmatch(r"\d{13}", r["video_id"].split("_", 1)[1]), r["video_id"]))
        for r in g:
            r["duplicate_of"] = "" if r is g[0] else g[0]["video_id"]

    reader = ClockReader.load(MODELS / "clock_knn.npz")

    def start_time(r):
        stem = r["video_id"].split("_", 1)[1]
        if r["duplicate_of"]:
            return r
        m = re.match(r"(\d{13})", stem)
        if m and not r["camera"].startswith("Camera1"):
            t = dt.datetime.fromtimestamp(int(m[1]) / 1000, TZ).replace(tzinfo=None)
            r.update(start_time=t.isoformat(sep=" "), start_source="filename")
        else:
            t, agree, n = reader.clip_start(RAW.parents[1] / r["path"])
            r.update(start_time=t.replace(microsecond=0).isoformat(sep=" ") if t else "",
                     start_source=f"clock_ocr ({agree}/{n} frames agree)")
        return r

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(start_time, rows))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    keys = ["video_id", "camera", "path", "start_time", "start_source", "duration_s", "fps", "width",
            "height", "codec", "md5", "duplicate_of"]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    u = [r for r in rows if not r["duplicate_of"]]
    print(f"wrote {MANIFEST}: {len(rows)} files, {len(u)} unique, {len(rows) - len(u)} duplicates")


if __name__ == "__main__":
    sys.exit(main())
