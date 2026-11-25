"""Render report/report.md to report/report.pdf (A4). Run from the repo root:
    python report/make_report.py
Needs the `markdown` package and Playwright's Chromium (for the PDF)."""
import pathlib, subprocess
import markdown

R = pathlib.Path(__file__).resolve().parent
md = (R / "report.md").read_text()
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
CSS = """
@page{size:A4;margin:16mm 16mm 18mm}
body{font-family:Inter,"DejaVu Sans",sans-serif;color:#1c2420;font-size:10pt;line-height:1.5;margin:0}
h1{font-size:21pt;line-height:1.2;margin:0 0 4pt}
h2{font-size:14pt;margin:18pt 0 6pt;padding-top:6pt;border-top:1.2pt solid #1c2420;break-after:avoid}
h3{font-size:11.5pt;margin:12pt 0 4pt;break-after:avoid}
p{margin:0 0 7pt} ul,ol{margin:0 0 8pt;padding-left:16pt} li{margin:0 0 3pt}
img{max-width:100%;max-height:82mm;width:auto;display:block;margin:6pt auto 2pt;break-inside:avoid}
p:has(> img){break-inside:avoid;margin:6pt 0 0}
em{color:#4d5852}
table{border-collapse:collapse;width:100%;font-size:8.6pt;margin:4pt 0 10pt;break-inside:avoid;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:3pt 5pt;border-bottom:.5pt solid #d6ddd8;vertical-align:top}
th{border-bottom:1pt solid #1c2420}
code{font-family:"DejaVu Sans Mono",monospace;font-size:8.6pt;background:#f1f4f2;padding:0 2pt;border-radius:2pt}
pre{background:#f1f4f2;padding:6pt 8pt;font-size:8pt;white-space:pre-wrap}
hr{border:0;border-top:.6pt solid #d6ddd8;margin:12pt 0}
"""
(R / "report.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Stray dog detection report</title>'
                               f'<style>{CSS}</style></head><body>{body}</body></html>')
js = f"""const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://{R / "report.html"}',{{waitUntil:'networkidle'}});
await p.pdf({{path:'{R / "report.pdf"}',format:'A4',printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',
footerTemplate:'<div style="font-size:7pt;color:#777;width:100%;text-align:center;font-family:Inter">Stray dog detection · technical report · <span class=pageNumber></span>/<span class=totalPages></span></div>',
margin:{{top:'16mm',bottom:'18mm',left:'16mm',right:'16mm'}}}});await b.close();}})()"""
subprocess.run(["node", "-e", js], check=True)
print("wrote report/report.pdf")
