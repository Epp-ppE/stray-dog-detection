"""Build the presentation as slides/slides.pdf (16:9) and an editable slides/slides.pptx with
speaker notes, from one slide list. Run from the repo root:  python slides/make_slides.py
Numbers for the statistics and recognition slides are read from outputs/ so they stay in sync.
"""
import html, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / "slides"
M = json.loads((ROOT / "outputs/eval/metrics.json").read_text())
FINAL = "yolo11m_960+animals"


def stats_slide():
    p = ROOT / "outputs/stats/summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    short = {"Camera1_LocationA": "Camera1", "Camera3_LocationC": "Camera3", "Camera4_LocationD": "Camera4"}
    rows = [["", *[short[c] for c in s]]]
    for key, lab, fmt in [("pct_clips_with_dogs", "Clips with dogs", "{:.0f}%"), ("visits", "Visits (gaps ≤ 2 min merged)", "{}"),
                          ("visits_per_day", "Visits per day", "{}"), ("max_dogs_at_once", "Most dogs at once", "{}"),
                          ("dog_minutes", "Dog-minutes on screen", "{:.0f}"), ("pct_dog_time_moving", "Time spent moving", "{:.0f}%")]:
        rows.append([lab, *[fmt.format(v[key]) for v in s.values()]])
    return dict(title="What the cameras saw (6–8 Jan 2025)", table=rows, image="outputs/stats/activity_by_hour.png",
                notes="These come from the full pipeline: the final detector on all 425 clips, then ByteTrack with settings tuned on the dev clips. "
                      "'Most dogs at once' is a lower bound on distinct dogs. Activity is concentrated in daylight around feeding times. "
                      "Most dog time is spent standing or eating rather than moving.")


def reid_slide():
    p = ROOT / "outputs/reid/summary.json"
    img = ROOT / "outputs/reid/group_02.jpg"
    if not p.exists() or not img.exists():
        return None
    r = json.loads(p.read_text())
    return dict(title="Recognising individual dogs: design + first demo",
                bullets=["Unit = a track: several crops of one dog, averaged",
                         "Free training signal: same track = same dog; two dogs in one frame = different dogs",
                         "Fine-tune a metric-learning model (triplet / ArcFace), then match with a time and place prior",
                         f"Demo, no training: {r['tracks_embedded']} Camera3 tracks → {r['groups']} candidate dogs",
                         "Hard parts: look-alike puppies, infrared, head-in-bowl views, fence"],
                image="outputs/reid/group_02.jpg", image_side=True,
                notes="Recognition is designed, and a small baseline is implemented. Without identity labels the groups are candidates, not verified identities. "
                      "The image shows one candidate group: the same white puppy with a dark mask in several clips. "
                      "The next step is to label 20–30 dogs across clips and train a proper re-identification model.")


def slides():
    b, f = M["yolo11s_640"], M[FINAL]
    out = [
        dict(kind="title", title="Finding stray dogs in raw camera footage",
             subtitle="Detection, tracking and visit statistics from three community cameras in Chiang Mai",
             foot="CMKL · Epp-ppE/stray-dog-detection",
             notes="Hello. This project builds a complete pipeline that finds and locates dogs in unlabelled camera footage, and turns the detections into information about visits and activity."),
        dict(title="Why count stray dogs?",
             bullets=["Rabies, road safety and animal welfare are real risks in Thai communities",
                      "Vaccination, sterilisation and feeding programmes need facts: how many dogs, where, when",
                      "CMKL students placed cameras at feeding spots: 447 clips, no labels",
                      "Goal: a complete pipeline that detects and locates dogs, plus useful statistics"],
             image="report/figures/contact_camera3.jpg", image_side=True,
             notes="Interventions like vaccination need basic numbers that are hard to get. The cameras give raw video but no labels, so everything has to work without training data."),
        dict(title="The data",
             table=[["Camera", "Scene", "Clips", "Footage", "Night"],
                    ["Camera1", "Fenced feeding area", "341", "117 min", "58"],
                    ["Camera3", "Water trough by a road", "73", "26 min", "7"],
                    ["Camera4", "Bench behind plants", "11", "4 min", "0"]],
             bullets=["2560×1440, 15 fps, motion-triggered clips of 3–30 s", "Night clips are black-and-white infrared"],
             image="report/figures/contact_camera1.jpg",
             notes="Three cameras, 2.4 hours of unique footage over three days. Camera1 dominates. The scenes are hard: wide-angle lens, a fence in front of Camera1, plants at Camera4 and infrared at night."),
        dict(title="Two hidden problems, fixed first",
             bullets=["22 of 447 files were exact duplicates (same fingerprint) → removed, 425 clips left",
                      "Camera1 filenames are export times, not recording times",
                      "Fix: read the clock burned into the video",
                      "Text recognition failed on bright sky, so a digit-by-digit reader was trained on Camera3/4 (true times known)",
                      "Held-out clips: 99.5% of digits right, all start times within 0.13 s"],
             image="report/figures/clock_variants.png",
             notes="Before any modelling I checked the data. Duplicates would have double-counted visits. Camera1's filenames claimed every clip came from one afternoon, but the overlay shows three days including nights. A small digit reader fixed that, and its accuracy was measured on clips it never saw."),
        dict(title="Pipeline (CPU only)", image="report/figures/pipeline.png", wide=True,
             bullets=["Sample 3 frames/s at 1280×720 · YOLO11 pretrained on COCO · every animal class = dog · ByteTrack · visits & activity"],
             notes="No GPU, so I sample 3 frames per second and shrink the frames. The detector is pretrained on COCO, because there are no labels to train on. ByteTrack links boxes into tracks, and short or weak tracks are dropped. Then tracks become visits and activity numbers."),
        dict(title="How it was evaluated",
             bullets=["180 frames chosen at random before seeing any model output, balanced by camera and day/night",
                      "Hand-labelled: 332 dogs, 28 'hard' (mostly hidden, ignored), 20 empty frames",
                      "Split by clip: dev (87 frames) to tune thresholds, test (93 frames) to report",
                      "Correct = box overlap (IoU) ≥ 0.5 · AP@0.5, precision, recall, F1",
                      "95% intervals from 1,000 bootstrap resamples · 9 setups compared"],
             notes="Because there were no labels, I made a small test set. The frames were picked at random before looking at results, so the test isn't tilted towards easy frames. Everything is tuned on dev and reported on test, so the numbers aren't flattered by tuning."),
        dict(title="Results: bigger model + every animal class = dog", image="report/figures/ap_bars.png", image_side=True,
             bullets=[f"Baseline YOLO11s @640: AP {b['test_AP50']:.2f}, F1 {b['test_F1']:.2f}",
                      f"Final YOLO11m @960 + animals: AP {f['test_AP50']:.2f} [{f['test_AP50_CI95'][0]:.2f}–{f['test_AP50_CI95'][1]:.2f}], F1 {f['test_F1']:.2f}",
                      f"Precision {f['test_precision']:.2f}, recall {f['test_recall']:.2f}",
                      "Chosen by dev AP; it also ranks first on test"],
             notes="A bigger model helps, but the biggest single gain is counting every animal class as dog. AP goes from 0.44 for the baseline to 0.75 for the final setup, with the interval shown."),
        dict(title="Why 'every animal = dog' works",
             bullets=["COCO has no 'puppy' and no infrared photos",
                      "28% of found dogs were labelled something else, mostly 'cat'",
                      "At night: 52% (cat, sheep, horse)",
                      "Night AP: 0.42 → 0.76 with all animal classes counted"],
             image="docs/img/puppy_as_cat.jpg", image_side=True,
             notes="This black puppy is called 'cat' with 92% confidence. In infrared, dogs become sheep or horses. Merging the animal classes, and removing duplicate boxes on the same animal, recovers them."),
        dict(title="Where it works, where it fails",
             table=[["Subset", "AP@0.5", "Recall"],
                    ["Camera1 (fence)", "0.70", "0.70"], ["Camera3 (road)", "0.82", "0.84"], ["Camera4 (plants)", "0.79", "0.80"],
                    ["Day", "0.77", "0.77"], ["Night (infrared)", "0.76", "0.76"],
                    ["Large dogs", "", "0.93"], ["Small dogs", "", "0.57"], ["Tiny dogs", "", "0.33"]],
             bullets=["Recall, not precision, is the limit", "Small and far-away dogs are the main misses"],
             notes="The fence at Camera1 makes it the hardest camera. Night is now on par with day. Size matters most: large dogs are found 93% of the time, tiny ones a third of the time."),
        dict(title="Successes", image="slides/success_grid.jpg", image_side=True,
             bullets=["(a) 6 dogs; 4 tiny ones on the far road found at confidence 0.02–0.30",
                      "(b) 4 dogs at night in infrared, including a pile of puppies",
                      "(c) dogs in plants and behind the bench, against the sun"],
             notes="Green boxes are my labels, yellow are correct detections. Note the confidences as low as 0.02 for the far dogs. That is why the tuned threshold is so low."),
        dict(title="Failures: false alarms", image="slides/false_alarms_grid.jpg", image_side=True,
             bullets=["(a) Pile of puppies: an extra box spanning two of them",
                      "(b) Part of a motorbike and rider, at confidence 0.02",
                      "(c) A dog lying in plants gets a second, shifted box",
                      "False alarms are rare: precision 0.81 on test"],
             notes="Red boxes are false alarms. Most come from groups of overlapping dogs. The motorbike box has very low confidence; it gets through only because the threshold is so low."),
        dict(title="Failures: missed dogs", image="slides/misses_grid.jpg", image_side=True,
             bullets=["(a) Dark puppies at the back, a dog behind the fence mesh",
                      "(b) Infrared glare washes out a dog at the bowl",
                      "(c) A black dog at the lens edge, pressed against others",
                      "Misses are the main error: recall 0.75 on test"],
             notes="Magenta boxes are dogs the model missed. Small, dark and partly hidden dogs are the hardest, especially behind Camera1's fence."),
    ]
    for extra in (stats_slide(), reid_slide()):
        if extra:
            out.append(extra)
    out += [
        dict(title="Limitations and next steps",
             bullets=["Small dogs, infrared glare, fence and plants, puppy piles",
                      "Confidence is badly calibrated: useful threshold ≈ 0.02, not 0.25",
                      "Only 3 days and 93 test frames: wide intervals",
                      "Finds dogs, not 'stray' dogs; counts are lower bounds",
                      "Next: fine-tune on local labels, tile frames for far dogs, collect weeks, label 20–30 dogs for recognition"],
             notes="The biggest remaining improvement would be fine-tuning on labelled frames from these cameras. My labels were kept for honest evaluation instead."),
        dict(kind="title", title="Thank you", subtitle="Code, report, labels and results: github.com/Epp-ppE/stray-dog-detection",
             foot="Everything reproduces with  bash scripts/run_all.sh", notes="Thank you. I'm happy to take questions."),
    ]
    return out


CSS = """
:root{--ink:#1c2420;--muted:#5b675f;--accent:#1f7a5a;--soft:#eef3f0;--line:#d6ddd8}
@page{size:1280px 720px;margin:0}
body{margin:0;font-family:Inter,"DejaVu Sans",sans-serif;color:var(--ink);background:#fff}
.slide{width:1280px;height:720px;box-sizing:border-box;padding:52px 64px 40px;position:relative;break-after:page;overflow:hidden;display:flex;flex-direction:column}
.slide h1{font-size:40px;margin:0 0 22px;letter-spacing:-.01em;line-height:1.15}
.slide .bar{position:absolute;left:0;top:0;width:12px;height:100%;background:var(--accent)}
.body{flex:1;display:flex;gap:36px;min-height:0}
.body.col{flex-direction:column;gap:14px}
.txt{flex:1;min-width:0}
ul{margin:0;padding-left:26px;font-size:23px;line-height:1.38}
li{margin:0 0 10px}
.img{flex:1.05;display:flex;align-items:center;justify-content:center;min-height:0}
.img img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}
.body.side .txt{flex:0.8}.body.side .img{flex:1.25}
.wide img{max-height:470px}
table{border-collapse:collapse;font-size:21px;margin:0 0 16px}
td,th{padding:6px 14px;border-bottom:1px solid var(--line);text-align:left}
th{border-bottom:2px solid var(--ink)}
.num{position:absolute;right:28px;bottom:18px;font-size:14px;color:var(--muted)}
.title{justify-content:center;padding-left:110px;background:var(--soft)}
.title h1{font-size:58px;max-width:900px}
.title p{font-size:26px;color:var(--muted);max-width:900px;margin:0 0 40px}
.title .foot{font-size:18px;color:var(--accent);font-weight:600}
.cap{font-size:20px;color:var(--muted);margin:0}
"""


def render_html(sl):
    parts = []
    for i, s in enumerate(sl, 1):
        if s.get("kind") == "title":
            parts.append(f'<section class="slide title"><div class="bar"></div><h1>{html.escape(s["title"])}</h1>'
                         f'<p>{html.escape(s["subtitle"])}</p><div class="foot">{html.escape(s.get("foot", ""))}</div></section>')
            continue
        tbl = ""
        if s.get("table"):
            rows = s["table"]
            tbl = "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in rows[0]) + "</tr>" + \
                  "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>"
        ul = "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in s.get("bullets", [])) + "</ul>" if s.get("bullets") else ""
        img = ""
        if s.get("image"):
            ims = f'<img src="../{s["image"]}">' + (f'<img src="../{s["image2"]}" style="margin-top:8px">' if s.get("image2") else "")
            img = f'<div class="img" style="flex-direction:column">{ims}</div>'
        if s.get("wide"):
            cap = f'<p class="cap">{html.escape(s["bullets"][0])}</p>' if s.get("bullets") else ""
            body = f'<div class="body col wide">{img}{cap}</div>'
        elif s.get("image_side") or (s.get("image") and not s.get("table")):
            body = f'<div class="body side"><div class="txt">{tbl}{ul}</div>{img}</div>'
        else:
            body = f'<div class="body"><div class="txt">{tbl}{ul}</div>{img}</div>'
        parts.append(f'<section class="slide"><div class="bar"></div><h1>{html.escape(s["title"])}</h1>{body}<div class="num">{i}</div></section>')
    return f'<!doctype html><html><head><meta charset="utf-8"><title>Slides</title><style>{CSS}</style></head><body>{"".join(parts)}</body></html>'


def render_pptx(sl):
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from PIL import Image
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    GREEN, INK, MUTED = RGBColor(0x1F, 0x7A, 0x5A), RGBColor(0x1C, 0x24, 0x20), RGBColor(0x5B, 0x67, 0x5F)

    def text(slide, x, y, w, h, lines, size, color=INK, bold=False, bullets=False):
        tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
        for k, line in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.text = ("• " if bullets else "") + line
            p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold; p.space_after = Pt(8)
        return tb

    def picture(slide, path, x, y, w, h):
        iw, ih = Image.open(ROOT / path).size
        s = min(w / iw, h / ih)
        pw, ph = int(iw * s), int(ih * s)
        slide.shapes.add_picture(str(ROOT / path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)

    for s in sl:
        slide = prs.slides.add_slide(blank)
        bar = slide.shapes.add_shape(1, 0, 0, Inches(0.12), prs.slide_height)
        bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()
        if s.get("kind") == "title":
            text(slide, Inches(1.1), Inches(2.3), Inches(10.5), Inches(1.6), [s["title"]], 44, bold=True)
            text(slide, Inches(1.1), Inches(3.9), Inches(10.5), Inches(1.0), [s["subtitle"]], 22, MUTED)
            text(slide, Inches(1.1), Inches(5.2), Inches(10.5), Inches(0.6), [s.get("foot", "")], 16, GREEN, bold=True)
        else:
            text(slide, Inches(0.6), Inches(0.35), Inches(12.2), Inches(0.9), [s["title"]], 32, bold=True)
            W, top, H = Emu(Inches(12.2)), Inches(1.35), Inches(5.8)
            if s.get("wide"):
                if s.get("image2"):
                    picture(slide, s["image"], Inches(0.6), top, W, Emu(int(H * 0.44)))
                    picture(slide, s["image2"], Inches(0.6), top + Emu(int(H * 0.46)), W, Emu(int(H * 0.44)))
                else:
                    picture(slide, s["image"], Inches(0.6), top, W, Emu(int(H * 0.85)))
                if s.get("bullets"):
                    text(slide, Inches(0.6), Inches(6.75), W, Inches(0.6), s["bullets"], 14, MUTED)
            else:
                tw = Inches(6.2) if s.get("image") else W
                y = top
                if s.get("table"):
                    rows = s["table"]
                    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(0.6), y, tw, Inches(0.38) * len(rows))
                    for r, row in enumerate(rows):
                        for c, v in enumerate(row):
                            cell = shp.table.cell(r, c); cell.text = v
                            cell.text_frame.paragraphs[0].font.size = Pt(14); cell.text_frame.paragraphs[0].font.bold = r == 0
                    y = y + Inches(0.4) * len(rows) + Inches(0.2)
                if s.get("bullets"):
                    text(slide, Inches(0.6), y, tw, Inches(5.5), s["bullets"], 20, bullets=True)
                if s.get("image"):
                    picture(slide, s["image"], Inches(7.0), top, Inches(5.8), H)
        slide.notes_slide.notes_text_frame.text = s.get("notes", "")
    prs.save(S / "slides.pptx")


if __name__ == "__main__":
    sl = slides()
    (S / "slides.html").write_text(render_html(sl))
    js = f"""const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://{S / "slides.html"}',{{waitUntil:'networkidle'}});
await p.pdf({{path:'{S / "slides.pdf"}',width:'1280px',height:'720px',printBackground:true}});await b.close();}})()"""
    subprocess.run(["node", "-e", js], check=True)
    render_pptx(sl)
    print(f"wrote slides.pdf and slides.pptx ({len(sl)} slides)")
