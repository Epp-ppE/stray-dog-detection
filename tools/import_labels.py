"""Merge labels exported from the labelling page's database (one JSON file per frame,
as written by an export of the 'labels' collection) into outputs/eval/labels.json.

usage: python tools/import_labels.py <export_dir>/labels
"""
import json, sys, pathlib

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "eval" / "labels.json"
labels = {}
for f in sorted(src.glob("*.json")):
    d = json.loads(f.read_text())
    d = d.get("data", d)
    labels[f.stem] = {"status": d["status"], "boxes": [
        {k: (bool(b[k]) if k == "hard" else round(float(b[k]), 5)) for k in ("x1", "y1", "x2", "y2", "hard")}
        for b in d.get("boxes", [])]}
out.write_text(json.dumps(labels, indent=1, sort_keys=True))
n_box = sum(len(v["boxes"]) for v in labels.values())
n_hard = sum(b["hard"] for v in labels.values() for b in v["boxes"])
st = {}
for v in labels.values():
    st[v["status"]] = st.get(v["status"], 0) + 1
print(f"{len(labels)} frames {st}; {n_box} boxes ({n_hard} hard) -> {out}")
