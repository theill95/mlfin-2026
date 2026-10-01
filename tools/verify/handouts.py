# -*- coding: utf-8 -*-
"""Check that every lecture PDF, the cheatsheet PDF and the maths PDF match their
current source, and that each lecture PDF has a page for every slide.

tools/build_handouts.py records the source it printed each PDF from; a deck
rendered again, or a page edited, without rebuilding its PDF fails here.

    python tools/verify/handouts.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
stamps = json.loads((ROOT / "downloads" / "lectures" / "built.json").read_text())
problems = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for html in sorted(ROOT.glob("session_*/session_*.html")):
    if not re.fullmatch(r"session_\d\d\.html", html.name):
        continue
    n = html.stem[-2:]
    pdf_path = ROOT / "downloads" / "lectures" / f"session_{n}.pdf"
    if not pdf_path.exists():
        problems.append(f"no PDF for session {int(n)}")
        continue
    if stamps.get(n) != sha(html):
        problems.append(f"session {int(n)}'s PDF is older than its deck")
    slides = len(re.findall(r'<section id="[^"]*" class="[^"]*\bslide level[12]', html.read_text(encoding="utf-8")))
    pages = pymupdf.open(pdf_path).page_count
    if pages < slides + 1:                 # + the title slide
        problems.append(f"session {int(n)}'s PDF has {pages} pages for {slides + 1} slides")
    print(f"session {int(n):2d}: {pages:3d} pages, {pdf_path.stat().st_size / 1e6:.1f} MB")

for key, src, pdf in [("cheatsheet", "cheatsheet.html", "cheatsheet.pdf"), ("maths", "math.html", "maths.pdf")]:
    if not (ROOT / "downloads" / pdf).exists():
        problems.append(f"no {pdf}")
    elif stamps.get(key) != sha(ROOT / src):
        problems.append(f"{pdf} is older than {src}")

if problems:
    print("\n".join(["", "PDFs out of date (python tools/build_handouts.py):"] + ["  - " + p for p in problems]))
    sys.exit(1)
print("every PDF matches its source")
