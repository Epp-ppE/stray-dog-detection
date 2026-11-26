"""Build the presentation as slides/slides.pdf (16:9) and an editable slides/slides.pptx with
speaker notes, from one slide list. Run from the repo root:  python slides/make_slides.py
Numbers are read from outputs/ where possible so they stay in sync with the results.
Slides after "Questions" are backup slides for answering questions.
"""
import html, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / "slides"
M = json.loads((ROOT / "outputs/eval/metrics.json").read_text())
ST = json.loads((ROOT / "outputs/stats/summary.json").read_text())
FINAL, BASE, TRACKED = "yolo11m_960+animals", "yolo11s_640", "yolo11m_960+animals+tracking"
CAMS = {"Camera1_LocationA": "Camera1", "Camera3_LocationC": "Camera3", "Camera4_LocationD": "Camera4"}


def ap(k, ci=False):
    m = M[k]
    s = f"{m['test_AP50']:.2f}"
    return s + (f" [{m['test_AP50_CI95'][0]:.2f}–{m['test_AP50_CI95'][1]:.2f}]" if ci else "")


def reid_pair():
    """Put the good and the mixed recognition group side by side, with a label above each."""
    from PIL import Image, ImageDraw, ImageFont
    a, b = (Image.open(ROOT / f"report/figures/{n}.jpg").convert("RGB") for n in ("reid_good", "reid_mixed"))
    gap, head = 40, 70
    out = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height) + head), "white")
    out.paste(a, (0, head)); out.paste(b, (a.width + gap, head))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except OSError:
        f = ImageFont.load_default()
    d.text((0, 14), "Looks right", fill=(31, 122, 90), font=f)
    d.text((a.width + gap, 14), "Mixes littermates", fill=(180, 84, 31), font=f)
    out.save(S / "reid_pair.jpg", quality=90)
    return "slides/reid_pair.jpg"


def slides():
    b, f, t = M[BASE], M[FINAL], M[TRACKED]
    br = M[FINAL]["breakdown_all_frames"]
    c1 = ST["Camera1_LocationA"]
    stats_rows = [["", *CAMS.values()]]
    for key, lab, fmt in [("pct_clips_with_dogs", "Clips with a dog", "{:.0f}%"),
                          ("visits_per_day", "Visits per day", "{:.0f}"),
                          ("median_visit_s", "Typical visit", "{:.0f} s"),
                          ("max_dogs_at_once", "Most dogs at once", "{}"),
                          ("dog_minutes", "Dog-minutes on screen", "{:.0f}"),
                          ("pct_dog_time_moving", "Time spent moving", "{:.0f}%"),
                          ("night_visits", "Night visits", "{}")]:
        stats_rows.append([lab, *[fmt.format(ST[c][key]) for c in CAMS]])

    def sub(k):
        return f"{br[k]['AP50']:.2f}", f"{br[k]['recall']:.2f}"

    return [
        dict(kind="title", title="Finding stray dogs in raw camera footage",
             subtitle="Detecting, tracking and counting dogs at three community feeding spots in Chiang Mai",
             foot="Phupa Denphatcharangkul · 01223049 · CMKL",
             notes="Hello, I'm Phupa. My project takes raw, unlabelled camera footage from feeding spots in Chiang Mai, "
                   "finds and locates every dog in it, and turns that into numbers about dog visits."),

        dict(title="Why count stray dogs?",
             lead="Dog programmes need basic numbers: how many dogs, where and when.",
             bullets=["Stray dogs bring risks: rabies, road accidents and poor welfare",
                      "Vaccination, sterilisation and feeding teams need to know where dogs are",
                      "Cameras at three feeding spots recorded the dogs, but nobody labelled the video",
                      "Goal: find and locate the dogs automatically, then turn that into visit statistics"],
             image="report/figures/contact_camera3.jpg",
             notes="Teams that vaccinate or sterilise dogs need to know how many dogs there are and when they come. "
                   "Cameras can collect that, but watching hours of video by hand doesn't scale. "
                   "So the goal is a pipeline that finds the dogs automatically."),

        dict(title="The data: 3 cameras, 3 days, no labels",
             table=[["Camera", "Scene", "Clips", "Video", "Night clips"],
                    ["Camera1", "Fenced feeding area", "341", "117 min", "58"],
                    ["Camera3", "Water trough by a road", "73", "26 min", "7"],
                    ["Camera4", "Bench behind plants", "11", "4 min", "0"]],
             bullets=["6–8 January 2025: short clips (3–30 s) recorded when the camera senses motion",
                      "Night clips are black-and-white infrared",
                      "Hard scenes: wide-angle lens, a fence, plants, glare"],
             image="report/figures/contact_camera1.jpg",
             notes="There are three cameras and about two and a half hours of unique video. Most of it comes from Camera1, "
                   "a fenced feeding area with many puppies. The scenes are difficult: Camera1 looks through a chain-link fence, "
                   "Camera4 is behind plants, and at night everything is infrared."),

        dict(title="Two hidden problems, fixed before modelling",
             bullets=["Duplicates: 22 of the 447 files were identical copies. I removed them (425 clips left), "
                      "otherwise visits would be counted twice",
                      "Wrong times: Camera1's filenames are export times. All clips looked like one afternoon, "
                      "but they really span three days and nights",
                      "Fix: read the clock printed on the video. Standard text recognition failed on bright sky, "
                      "so I trained a small digit reader on Camera3 and 4, whose filenames are correct",
                      "Tested on unseen clips: 99.5% of digits right, every start time within 0.13 s"],
             image="report/figures/clock_variants.png",
             notes="Before building anything I checked the data. Duplicates were found by comparing file fingerprints. "
                   "The bigger problem was Camera1's timestamps: they were wrong, which would ruin any time-of-day statistic. "
                   "I fixed them by reading the on-screen clock digit by digit, and I measured how accurate that is on clips the reader never saw."),

        dict(title="The pipeline (runs on a normal CPU)", image="report/figures/pipeline.png", wide=True,
             bullets=["3 frames per second, resized to 1280×720  →  YOLO11 detector, pretrained on COCO  →  "
                      "every animal class counted as dog  →  ByteTrack links boxes into tracks  →  visits and activity"],
             notes="I sample three frames per second, which is enough to follow a walking dog and five times less work. "
                   "The detector is YOLO11, pretrained on the public COCO dataset. I didn't train it, because there were no labels. "
                   "Every animal class is counted as a dog, for a reason I'll show in a moment. "
                   "ByteTrack then follows each dog through a clip, and the tracks become visits and activity numbers."),

        dict(title="How I tested it",
             lead="A fair test: frames picked before seeing any results, tuning and scoring kept apart.",
             bullets=["180 frames picked at random, balanced across cameras and day/night",
                      "I labelled them by hand: 332 dogs, 28 'hard' dogs (mostly hidden, not scored), 20 empty frames",
                      "Split by clip: dev half (87 frames) to tune settings, test half (93 frames) to report",
                      "A detection is correct if its box overlaps a labelled dog by IoU ≥ 0.5",
                      "Scores: AP@0.5, precision, recall, F1, with 95% confidence intervals"],
             notes="There were no labels, so I made a test set. The computer picked the frames at random before I looked at any results, "
                   "so the test isn't biased towards easy frames. Every setting was chosen on the dev half, and every number I show is on "
                   "the test half, which was never used for tuning. Precision means how many boxes are real dogs, recall means how many dogs were found."),

        dict(title=f"Result: AP {ap(BASE)} → {ap(FINAL)}",
             kpis=[(ap(BASE), "baseline: YOLO11s, 640 px, 'dog' class only"),
                   (ap(FINAL), "final: YOLO11m, 960 px, every animal = dog")],
             table=[["Setup (test half)", "AP@0.5"],
                    ["Baseline", ap(BASE)],
                    ["Bigger model (m)", ap("yolo11m_640")],
                    ["Bigger model + bigger input", ap("yolo11m_960")],
                    ["+ every animal counted as dog", ap(FINAL, ci=True)]],
             hl=[4],
             image="report/figures/ap_bars.png",
             notes=f"The baseline is the obvious default: a small YOLO model, looking only for the dog class. It scores {ap(BASE)}. "
                   f"A bigger model helps a lot; a bigger input image helps only a little on its own. "
                   f"The biggest jump comes from counting every animal class as a dog, which brings it to {ap(FINAL)}. "
                   f"F1 goes from {b['test_F1']:.2f} to {f['test_F1']:.2f}. I chose this setup on the dev half, and it is also the best on test."),

        dict(title="Why count every animal as a dog?",
             lead="The detector knows dogs, but not puppies or infrared.",
             bullets=["COCO, the dataset it learned from, has no puppies and no infrared images",
                      "28% of the dogs it found had another top label, mostly 'cat'",
                      "At night it was 52%: 'sheep', 'horse' or 'cow'",
                      "Counting all animal classes as dog: +0.17 to +0.20 AP, night AP 0.42 → 0.76",
                      "No real cats or livestock appear in these scenes, so it costs almost nothing here"],
             image="slides/puppy_as_cat.jpg",
             notes="This black puppy is detected with 92% confidence, but as a cat. In infrared, dogs often come out as sheep or cows. "
                   "So the model already finds these dogs; it just names them wrongly. Counting every animal class as a dog, "
                   "and removing duplicate boxes on the same animal, recovers them. Near farms this would need checking, but not here."),

        dict(title="Where it works and where it fails",
             table=[["Subset", "AP@0.5", "Recall"],
                    ["Camera1 (fence)", *sub("camera=Camera1_LocationA")],
                    ["Camera3 (road)", *sub("camera=Camera3_LocationC")],
                    ["Camera4 (plants)", *sub("camera=Camera4_LocationD")],
                    ["Day", *sub("night=False")], ["Night (infrared)", *sub("night=True")]],
             bullets=["Night is now as good as day",
                      "The fence makes Camera1 the hardest camera",
                      "Size matters most: large dogs 93% found, tiny dogs 33%",
                      "Most errors are missed dogs, not false alarms"],
             notes="Broken down by condition: Camera1, behind the fence, is hardest. Night is now on par with day. "
                   "The biggest factor is size: large dogs are almost always found, tiny far-away dogs only a third of the time."),

        dict(title="Successes", image="slides/success_grid.jpg", image_side=True,
             bullets=["Four tiny dogs on the far road, found at confidence 0.02–0.30",
                      "Four dogs at night in infrared, including a pile of puppies",
                      "Dogs in plants, against the sun"],
             notes="Green boxes are my labels and yellow boxes are correct detections. "
                   "Notice how low some confidences are, 0.02 for the far dogs. That's why the tuned threshold is so low."),

        dict(title="Failures: false alarms", image="slides/false_alarms_grid.jpg", image_side=True,
             bullets=["An extra box spanning two puppies",
                      "Part of a motorbike and its rider",
                      "A second, shifted box on a dog lying in plants",
                      f"False alarms are rare: precision {f['test_precision']:.2f}"],
             notes="Red boxes are false alarms. Most come from groups of dogs overlapping each other. "
                   "The motorbike box has a very low confidence and only gets through because the threshold is low."),

        dict(title="Failures: missed dogs", image="slides/misses_grid.jpg", image_side=True,
             bullets=["Dark puppies at the back, a dog behind the fence mesh",
                      "Infrared glare washing out a dog at the bowl",
                      "A black dog cut off at the lens edge",
                      f"Missed dogs are the main error: recall {f['test_recall']:.2f}"],
             notes="Magenta boxes are dogs the model missed. Small, dark and partly hidden dogs are the hardest, "
                   "especially behind Camera1's fence."),

        dict(title="Tracking: fewer dogs, but real ones",
             kpis=[(f"{f['test_precision']:.2f} → {t['test_precision']:.2f}", "precision: share of boxes that are real dogs"),
                   (f"{f['test_recall']:.2f} → {t['test_recall']:.2f}", "recall: share of dogs found")],
             bullets=["ByteTrack follows each dog through a clip. A track counts only if it lasts ≥ 2 frames "
                      "and is confident at least once (settings tuned on the dev half)",
                      "For statistics, a fake visit is worse than a missed frame, so precision comes first",
                      "So the statistics are lower bounds: the dogs counted are real, but some are missed",
                      "Full run: 425 clips, 26,325 frames, 2.7 hours on 4 CPU cores → 1,073 dog tracks"],
             notes="Tracking removes boxes that flicker for a single frame. That makes 95% of the remaining boxes real dogs, "
                   "but it drops about half of them, mostly brief and small ones. For counting visits that's the right trade, "
                   "because a false track would invent a visit. It means the numbers on the next slide are conservative."),

        dict(title="What the cameras saw (6–8 Jan 2025)",
             lead=f"Camera1's feeding area: about {c1['visits_per_day']:.0f} dog visits a day, mostly 06:00–18:00.",
             table=stats_rows, image="outputs/stats/activity_by_hour.png",
             notes="Almost every clip contains a dog, so the motion trigger is a good record of dog presence. "
                   "Camera1 is busiest, with visits all day and some at night. Camera3's peak of seven dogs was a litter of puppies. "
                   "Dogs move only 11 to 23% of the time: they mostly stand, eat and drink. "
                   "One caution: this is only three days, and the cameras record only when triggered."),

        dict(title="Recognising individual dogs: the design",
             lead="Detection says 'a dog is here'. Recognition says 'this is dog #12, seen yesterday'.",
             bullets=["Unit = a track: several pictures of one dog, averaged",
                      "Free training data: same track = same dog; two dogs in one frame = different dogs",
                      "Model: fine-tune an embedding with metric learning (triplet or ArcFace loss)",
                      "Matching: appearance + time and place; a new dog if nothing is close; a person checks unsure matches",
                      "Evaluate on 20–30 hand-matched dogs: rank-1 accuracy and error in the dog count"],
             notes="Recognition would turn visit counts into the number of distinct dogs. The tracker already gives many pictures of the "
                   "same dog for free, and two dogs in the same frame must be different, which gives training pairs without labelling. "
                   "Matching combines appearance with time and place, and a person confirms uncertain matches. "
                   "Night needs a separate, stricter matching because colour disappears."),

        dict(title="Recognition demo, without any training",
             bullets=["Generic image features + colour, then grouping similar tracks",
                      "150 Camera3 tracks → 72 candidate dogs, 30 seen in more than one clip",
                      "Left: the same puppy in 6 clips over 3 days",
                      "Right: look-alike littermates wrongly merged",
                      "No identity labels yet, so these are candidates for a person to check"],
             image=reid_pair(), image_side=True,
             notes="As a first test I grouped Camera3's tracks using generic image features, with no training at all. "
                   "Some groups look right, like this puppy seen across three days. Others merge siblings that look almost the same. "
                   "That's exactly what the fine-tuned model in the design is meant to fix."),

        dict(title="Limitations and next steps",
             table=[["Limitation", "Next step"],
                    ["Small, far-away dogs (tiny: 33% found)", "Split frames into tiles, or a GPU at higher resolution"],
                    ["Puppies, infrared, the fence: unlike COCO", "Fine-tune on a few hundred local frames"],
                    ["Low confidence: useful threshold ≈ 0.02, not 0.25", "Re-check it for new cameras or seasons"],
                    ["Only 93 test frames and 3 days", "Collect weeks of footage, label more"],
                    ["Finds dogs, not 'stray' dogs; counts are lower bounds", "Recognition; a camera at dog eye level"]],
             notes="The main limits are small dogs, the scenes being very different from what the model was trained on, and a small test set. "
                   "The single biggest improvement would be fine-tuning on labelled frames from these cameras. "
                   "I kept my labels for honest testing instead."),

        dict(kind="title", title="Thank you. Questions?",
             subtitle="Code, report, labels and results: github.com/Epp-ppE/stray-dog-detection",
             foot="Everything reproduces with  bash scripts/run_all.sh",
             notes="Thank you. I'm happy to take questions."),

        # ---------------- backup slides ----------------
        dict(title="Backup: all setups (test half)", backup=True,
             table=[["Setup", "AP@0.5 [95% CI]", "Precision", "Recall", "F1"]] + [
                 [lab, ap(k, ci=True), f"{M[k]['test_precision']:.2f}", f"{M[k]['test_recall']:.2f}", f"{M[k]['test_F1']:.2f}"]
                 for k, lab in [("yolo11n_640", "YOLO11n @640"), ("yolo11s_640", "YOLO11s @640 (baseline)"),
                                ("yolo11m_640", "YOLO11m @640"), ("yolo11s_960", "YOLO11s @960"), ("yolo11m_960", "YOLO11m @960"),
                                ("yolo11s_640+animals", "YOLO11s @640 + animals"), ("yolo11m_640+animals", "YOLO11m @640 + animals"),
                                (FINAL, "YOLO11m @960 + animals (final)"), (TRACKED, "Final + tracking (tuned)")]],
             hl=[8],
             notes="All setups, scored on the test half at thresholds chosen on dev. Neighbouring setups overlap in their intervals; "
                   "the clear gains are the bigger model and counting all animals."),

        dict(title="Backup: settings and definitions", backup=True,
             table=[["Item", "Value"],
                    ["Confidence threshold (final detector)", f"{f['threshold_from_dev']:.3f}, chosen on dev by best F1"],
                    ["At the usual 0.25 threshold", f"precision {f['at_default_0_25']['precision']:.2f}, recall {f['at_default_0_25']['recall']:.2f}"],
                    ["Duplicate boxes on one animal", "removed if overlap (IoU) > 0.6"],
                    ["Tracker: start / match a track", "confidence ≥ 0.10; second-stage match ≥ 0.01"],
                    ["Confirmed track", "≥ 2 frames and peak confidence ≥ 0.10"],
                    ["Visit", "clips with dogs less than 2 min apart, merged"],
                    ["Moving", "faster than 0.25 body lengths per second"],
                    ["Close contact", "centres closer than the dogs' average box width"]],
             notes="Details in case someone asks how a setting or a statistic is defined."),

        dict(title="Backup: running it", backup=True,
             table=[["Step", "Time (4-core CPU)", "Needs the video?"],
                    ["Everything: bash scripts/run_all.sh", "about 3 h", "yes (2.4 GB)"],
                    ["Detection on all clips", "2.7 h", "yes"],
                    ["Re-score from saved predictions: python evaluate.py", "about 10 min", "no"],
                    ["Statistics: python stats.py", "under 1 min", "no"],
                    ["Recognition demo: python reid_demo.py", "about 2 min", "yes"]],
             bullets=["Labels, evaluation frames, cached predictions, tracks and statistics are all in the repository",
                      "The evaluation was re-run on a second computer and gave the same numbers"],
             notes="If asked about reproducibility: the expensive step is detection, but the evaluation reproduces in minutes "
                   "from the committed predictions, without downloading the video."),
    ]


CSS = """
:root{--ink:#1c2420;--muted:#5b675f;--accent:#1f7a5a;--soft:#eef3f0;--line:#d6ddd8;--warn:#b4541f}
@page{size:1280px 720px;margin:0}
body{margin:0;font-family:Inter,"DejaVu Sans",sans-serif;color:var(--ink);background:#fff}
.slide{width:1280px;height:720px;box-sizing:border-box;padding:48px 64px 40px;position:relative;break-after:page;overflow:hidden;display:flex;flex-direction:column}
.slide h1{font-size:38px;margin:0 0 10px;letter-spacing:-.01em;line-height:1.15}
.lead{font-size:22px;color:var(--accent);font-weight:600;margin:0 0 14px}
.slide .bar{position:absolute;left:0;top:0;width:12px;height:100%;background:var(--accent)}
.backup .bar{background:var(--line)}
.tag{position:absolute;right:28px;top:22px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.body{flex:1;display:flex;gap:36px;min-height:0;margin-top:10px}
.body.col{flex-direction:column;gap:14px}
.txt{flex:1;min-width:0}
ul{margin:0;padding-left:26px;font-size:22px;line-height:1.36}
li{margin:0 0 10px}
.img{flex:1.05;display:flex;align-items:center;justify-content:center;min-height:0}
.img img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}
.body.side .txt{flex:0.8}.body.side .img{flex:1.25}
.wide img{max-height:470px}
table{border-collapse:collapse;font-size:20px;margin:0 0 16px;font-variant-numeric:tabular-nums}
td,th{padding:6px 14px;border-bottom:1px solid var(--line);text-align:left}
th{border-bottom:2px solid var(--ink)}
tr.hl td{font-weight:700;color:var(--accent)}
table.small{font-size:16px}
.solo ul{font-size:27px;line-height:1.4} .solo li{margin-bottom:16px}
.solo table{width:100%;font-size:23px} .solo table.small{font-size:19px} .solo td,.solo th{padding:9px 14px}
.kpis{display:flex;gap:28px;margin:0 0 18px}
.kpi{flex:1;background:var(--soft);border-radius:6px;padding:14px 20px}
.kpi b{display:block;font-size:46px;color:var(--accent);line-height:1.1}
.kpi span{font-size:17px;color:var(--muted)}
.legend{font-size:16px;color:var(--muted);margin-top:6px}
.num{position:absolute;right:28px;bottom:18px;font-size:14px;color:var(--muted)}
.title{justify-content:center;padding-left:110px;background:var(--soft)}
.title h1{font-size:56px;max-width:950px;margin-bottom:18px}
.title p{font-size:26px;color:var(--muted);max-width:900px;margin:0 0 40px}
.title .foot{font-size:20px;color:var(--accent);font-weight:600}
.cap{font-size:20px;color:var(--muted);margin:0}
"""


def render_html(sl):
    e = html.escape
    parts = []
    for i, s in enumerate(sl, 1):
        if s.get("kind") == "title":
            parts.append(f'<section class="slide title"><div class="bar"></div><h1>{e(s["title"])}</h1>'
                         f'<p>{e(s["subtitle"])}</p><div class="foot">{e(s.get("foot", ""))}</div></section>')
            continue
        tbl = ""
        if s.get("table"):
            rows, hl = s["table"], set(s.get("hl", []))
            cls = ' class="small"' if len(rows) > 8 or len(rows[0]) > 5 else ""
            tbl = f"<table{cls}><tr>" + "".join(f"<th>{e(c)}</th>" for c in rows[0]) + "</tr>" + "".join(
                f'<tr{" class=hl" if r in hl else ""}>' + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>"
                for r, row in enumerate(rows[1:], 1)) + "</table>"
        ul = "<ul>" + "".join(f"<li>{e(x)}</li>" for x in s.get("bullets", [])) + "</ul>" if s.get("bullets") else ""
        kp = ('<div class="kpis">' + "".join(f"<div class=kpi><b>{e(v)}</b><span>{e(l)}</span></div>" for v, l in s["kpis"])
              + "</div>") if s.get("kpis") else ""
        lead = f'<p class="lead">{e(s["lead"])}</p>' if s.get("lead") else ""
        leg = f'<p class="legend">{e(s["legend"])}</p>' if s.get("legend") else ""
        img = f'<div class="img" style="flex-direction:column"><img src="../{s["image"]}">{leg}</div>' if s.get("image") else ""
        if s.get("wide"):
            cap = f'<p class="cap">{e(s["bullets"][0])}</p>' if s.get("bullets") else ""
            body = f'<div class="body col wide">{img}{cap}</div>'
        elif s.get("image_side") or (s.get("image") and not s.get("table")):
            body = f'<div class="body side"><div class="txt">{tbl}{ul}</div>{img}</div>'
        else:
            solo = "" if s.get("image") else " solo"
            body = f'<div class="body{solo}"><div class="txt">{tbl}{ul}</div>{img}</div>'
        bk = " backup" if s.get("backup") else ""
        tag = '<div class="tag">backup</div>' if s.get("backup") else ""
        parts.append(f'<section class="slide{bk}"><div class="bar"></div>{tag}<h1>{e(s["title"])}</h1>{lead}{kp}{body}'
                     f'<div class="num">{i}</div></section>')
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
    SOFT, GREY = RGBColor(0xEE, 0xF3, 0xF0), RGBColor(0xD6, 0xDD, 0xD8)

    def text(slide, x, y, w, h, lines, size, color=INK, bold=False, bullets=False):
        tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
        for k, line in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.text = ("• " if bullets else "") + line
            p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold; p.space_after = Pt(8)
        return tb

    def picture(slide, path, x, y, w, h):
        iw, ih = Image.open(ROOT / path).size
        sc = min(w / iw, h / ih)
        pw, ph = int(iw * sc), int(ih * sc)
        slide.shapes.add_picture(str(ROOT / path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)

    for s in sl:
        slide = prs.slides.add_slide(blank)
        bar = slide.shapes.add_shape(1, 0, 0, Inches(0.12), prs.slide_height)
        bar.fill.solid(); bar.fill.fore_color.rgb = GREY if s.get("backup") else GREEN; bar.line.fill.background()
        if s.get("kind") == "title":
            slide.background.fill.solid(); slide.background.fill.fore_color.rgb = SOFT
            text(slide, Inches(1.1), Inches(2.2), Inches(10.5), Inches(1.6), [s["title"]], 44, bold=True)
            text(slide, Inches(1.1), Inches(3.9), Inches(10.5), Inches(1.0), [s["subtitle"]], 22, MUTED)
            text(slide, Inches(1.1), Inches(5.2), Inches(10.5), Inches(0.6), [s.get("foot", "")], 16, GREEN, bold=True)
            slide.notes_slide.notes_text_frame.text = s.get("notes", "")
            continue
        text(slide, Inches(0.6), Inches(0.3), Inches(12.2), Inches(0.8), [s["title"]], 30, bold=True)
        if s.get("backup"):
            text(slide, Inches(11.6), Inches(0.1), Inches(1.5), Inches(0.4), ["BACKUP"], 11, MUTED)
        y = Inches(1.05)
        if s.get("lead"):
            text(slide, Inches(0.6), y, Inches(12.2), Inches(0.5), [s["lead"]], 18, GREEN, bold=True)
            y += Inches(0.6)
        if s.get("kpis"):
            n = len(s["kpis"]); kw = (Inches(12.2) - Inches(0.3) * (n - 1)) // n
            for k, (v, lab) in enumerate(s["kpis"]):
                x = Inches(0.6) + k * (kw + Inches(0.3))
                box = slide.shapes.add_shape(1, x, y, kw, Inches(1.15))
                box.fill.solid(); box.fill.fore_color.rgb = SOFT; box.line.fill.background()
                text(slide, x + Inches(0.15), y + Inches(0.02), kw, Inches(0.7), [v], 36, GREEN, bold=True)
                text(slide, x + Inches(0.15), y + Inches(0.68), kw, Inches(0.4), [lab], 13, MUTED)
            y += Inches(1.35)
        top, H = y, Inches(7.1) - y
        W = Emu(Inches(12.2))
        if s.get("wide"):
            picture(slide, s["image"], Inches(0.6), top, W, Emu(int(H * 0.82)))
            if s.get("bullets"):
                text(slide, Inches(0.6), Inches(6.5), W, Inches(0.7), s["bullets"], 15, MUTED)
        else:
            side = s.get("image_side") or (s.get("image") and not s.get("table"))
            tw = Inches(5.3) if side and s.get("image") else (Inches(6.4) if s.get("image") else W)
            if s.get("table"):
                rows, hl = s["table"], set(s.get("hl", []))
                small = len(rows) > 8 or len(rows[0]) > 5
                solo = not s.get("image")
                rh = Inches(0.36 if small else (0.5 if solo else 0.42))
                shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(0.6), y, tw, rh * len(rows))
                for r, row in enumerate(rows):
                    for c, v in enumerate(row):
                        cell = shp.table.cell(r, c); cell.text = v
                        p = cell.text_frame.paragraphs[0]
                        p.font.size = Pt(12 if small else (18 if solo else 14)); p.font.bold = r == 0 or r in hl
                # estimate wrapped lines per row so text below the table doesn't overlap it
                pt = 12 if small else (18 if solo else 14)
                per_col = tw / len(rows[0]) / 914400 * 72 / (pt * 0.55)   # characters per line in a column
                lines = [max(1, max(-(-len(v) // max(1, int(per_col))) for v in row)) for row in rows]
                y = y + sum(rh * n for n in lines) + Inches(0.25)
            if s.get("bullets"):
                text(slide, Inches(0.6), y, tw, Inches(7.1) - y, s["bullets"], 18 if s.get("image") else 22, bullets=True)
            if s.get("image"):
                ix = Inches(0.6) + tw + Inches(0.4)
                iw = Inches(12.8) - ix
                picture(slide, s["image"], ix, top, iw, H - (Inches(0.4) if s.get("legend") else 0))
                if s.get("legend"):
                    text(slide, ix, Inches(6.75), iw, Inches(0.4), [s["legend"]], 13, MUTED)
        slide.notes_slide.notes_text_frame.text = s.get("notes", "")
    prs.save(S / "slides.pptx")


if __name__ == "__main__":
    sl = slides()
    (S / "slides.html").write_text(render_html(sl))
    js = f"""const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();const p=await b.newPage();
await p.goto('{(S / "slides.html").as_uri()}',{{waitUntil:'networkidle'}});
await p.pdf({{path:'{(S / "slides.pdf").as_posix()}',width:'1280px',height:'720px',printBackground:true}});await b.close();}})()"""
    subprocess.run(["node", "-e", js], check=True)
    render_pptx(sl)
    print(f"wrote slides.pdf and slides.pptx ({len(sl)} slides)")
