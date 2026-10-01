# -*- coding: utf-8 -*-
"""Build the PDFs students take into the exam: every lecture, the cheatsheet and
the maths brush-up.

A lecture printed from the browser loses its live cells' output, because a cell
only runs once its slide is on screen and both ways of printing a Reveal deck
(Quarto's `e` reloads the page in Reveal's PDF view, ?print-pdf) start the deck
again. So this script opens each deck in that PDF view, scrolls every slide into view, waits until every
cell that runs by itself has run, and only then prints the whole deck in one go:
one page per slide, real text a PDF reader can search, the fonts stored once.

A few slides build up in steps that replace each other (a click-through diagram,
a loop traced line by line). Printed in one go they would show every step on top
of the others, so those slides are printed again in the ordinary view, one page
per step, and spliced in where the slide was. The PDF gets an outline of the
slide titles.

    python tools/build_handouts.py              everything whose source changed
    python tools/build_handouts.py --all        everything
    python tools/build_handouts.py --session 6  one deck (repeatable)

Needs tools/requirements-dev.txt (Playwright, PyMuPDF) and Microsoft Edge, which
Playwright drives directly, so `playwright install` is not needed. Render the
decks first. Writes downloads/lectures/session_NN.pdf, downloads/cheatsheet.pdf
and downloads/maths.pdf, and records the source each was built from in
downloads/lectures/built.json, so an unchanged deck is not printed again.
"""
import argparse
import base64
import functools
import hashlib
import http.server
import json
import re
import sys
import threading
import time
from pathlib import Path

import pymupdf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
LECTURES = ROOT / "downloads" / "lectures"
STAMP = LECTURES / "built.json"
SITE_PDFS = {"cheatsheet": ("cheatsheet.html", ROOT / "downloads" / "cheatsheet.pdf"),
             "maths": ("math.html", ROOT / "downloads" / "maths.pdf")}
SCRIPT = re.compile(r'<script type="pyodide-(\d+)-contents">\s*([^<]+?)\s*</script>')
STEP_KINDS = ("current-visible", "fade-out", "semi-fade-out", "fade-in-then-out",
              "fade-in-then-semi-out")

# Hidden on paper: the menu button, the progress bar, the arrows, the slide and
# page counters (the PDF numbers its own pages), the editors' buttons, which do
# nothing on a page, and the note the page shows while Python is downloading.
PRINT_CSS = """
.slide-menu-button, .reveal .progress, .reveal .controls, .reveal .slide-number,
.reveal .slide-number-pdf,
.btn-group-exercise-editor, #exercise-loading-indicator, #exercise-loading-status
{ display: none !important; }
"""

# Every cell that runs by itself has run, and none is running now. (A hidden
# helper block keeps the class ojs-in-a-box-waiting-for-module-import after its
# cell has run, so that class is no signal; the spinner and the output are.)
CELLS_STATE = """(autorun) => {
  const cells = [...document.querySelectorAll('.exercise-cell')];
  if (cells.length !== autorun.length) return 'count ' + cells.length + ' of ' + autorun.length;
  const busy = [...document.querySelectorAll('.exercise-editor-eval-indicator')]
    .some((e) => !e.classList.contains('d-none'));
  if (busy) return 'busy';
  const missing = cells.map((c, i) => autorun[i] &&
      !c.querySelector('.cell-output-container, .exercise-cell-output') ? i : -1).filter((i) => i >= 0);
  if (missing.length) return 'missing ' + missing.join(',');
  return [...document.images].every((im) => im.complete) ? 'ok' : 'images';
}"""

SLIDES = """(stepKinds) => Reveal.getSlides().map((s) => {
  const ix = Reveal.getIndices(s);
  const h = s.querySelector('h1, h2');
  const frags = [...s.querySelectorAll('.fragment')];
  const stepped = frags.some((f) => stepKinds.some((k) => f.classList.contains(k)));
  const indices = frags.map((f) => +f.getAttribute('data-fragment-index'));
  return { h: ix.h, v: ix.v || 0, title: (h ? h.textContent : '').replace(/\\s+/g, ' ').trim(),
           steps: stepped ? Math.max(...indices) + 1 : 0, cells: s.querySelectorAll('.exercise-cell').length };
})"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve():
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def autorun_flags(html):
    """The autorun setting of each live cell, in the order the cells appear."""
    cells = sorted((int(n), json.loads(base64.b64decode(p))) for n, p in SCRIPT.findall(html))
    return [bool(block["attr"].get("autorun", True)) for _, block in cells]


def open_deck(browser, url, media):
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    page = context.new_page()
    page.emulate_media(media=media)
    page.goto(url, wait_until="load")
    page.wait_for_function("() => window.Reveal && Reveal.isReady()", timeout=60_000)
    page.add_style_tag(content=PRINT_CSS)
    if "print-pdf" not in url:
        page.evaluate("() => Reveal.configure({ transition: 'none', backgroundTransition: 'none' })")
    return context, page


def wait_for_cells(page, autorun, nudge=False, timeout=300):
    """Wait until every autorun cell has run; in the PDF view, nudge the missing ones into view."""
    t0 = time.time()
    while True:
        state = page.evaluate(CELLS_STATE, autorun)
        if state == "ok":
            return
        if time.time() - t0 > timeout:
            raise RuntimeError(f"cells did not finish: {state}")
        if nudge and state.startswith("missing"):
            for j in [int(x) for x in state.split()[1].split(",")][:3]:
                page.evaluate("(j) => { const c = document.querySelectorAll('.exercise-cell')[j]; if (c) c.scrollIntoView(); }", j)
        page.wait_for_timeout(500)


def step_pages(browser, url, slides, autorun):
    """The slides that build up in replacing steps, one page per step, in the ordinary view."""
    pages = {}
    todo = [k for k, s in enumerate(slides) if s["steps"]]
    if not todo:
        return pages
    context, page = open_deck(browser, url, "screen")
    for k in todo:
        s = slides[k]
        page.evaluate("([h, v]) => Reveal.slide(h, v)", [s["h"], s["v"]])
        if s["cells"]:
            wait_for_cells(page, autorun)
        doc = pymupdf.open()
        for f in range(-1, s["steps"]):
            page.evaluate("([h, v, f]) => Reveal.slide(h, v, f)", [s["h"], s["v"], f])
            page.wait_for_timeout(300)
            doc.insert_pdf(pymupdf.open("pdf", page.pdf(width="1280px", height="720px",
                                                         print_background=True, page_ranges="1")))
        pages[k] = doc
    context.close()
    return pages


def build_deck(browser, base_url, n):
    html_path = ROOT / f"session_{n:02d}" / f"session_{n:02d}.html"
    html = html_path.read_text(encoding="utf-8")
    autorun = autorun_flags(html)
    url = f"{base_url}/session_{n:02d}/session_{n:02d}.html"
    t0 = time.time()

    # The slides, read in the ordinary view.
    context, page = open_deck(browser, url, "screen")
    slides = page.evaluate(SLIDES, list(STEP_KINDS))
    context.close()

    # The whole deck, printed once from Reveal's PDF view (what Quarto's `e` reloads the
    # page into) after every cell has run: one page per slide, laid out for print.
    context, page = open_deck(browser, url + "?print-pdf", "print")
    page.wait_for_function("() => document.documentElement.classList.contains('print-pdf')", timeout=10_000)
    page.wait_for_timeout(1500)                    # the view lays the slides out
    i = 0
    while page.evaluate("""(i) => { const s = document.querySelectorAll('.reveal .slides section');
                                       if (i >= s.length) return false;
                                       s[i].scrollIntoView(); return true; }""", i):
        i += 1
        page.wait_for_timeout(120)
    wait_for_cells(page, autorun, nudge=True)
    page.wait_for_timeout(1500)                    # let the last figures settle
    whole = pymupdf.open("pdf", page.pdf(print_background=True, prefer_css_page_size=True))
    context.close()
    if whole.page_count != len(slides):
        raise RuntimeError(f"session {n}: {whole.page_count} pages for {len(slides)} slides")

    # The stepped slides again, one page per step, spliced in.
    steps = step_pages(browser, url, slides, autorun)
    pdf, toc = pymupdf.open(), []
    for k, s in enumerate(slides):
        if s["title"]:
            toc.append([1, s["title"], pdf.page_count + 1])
        if k in steps:
            pdf.insert_pdf(steps[k])
        else:
            pdf.insert_pdf(whole, from_page=k, to_page=k)
    title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", html, re.S).group(1)).strip()
    pdf.set_metadata({"title": f"Session {n}: {title}", "author": "Jonas Theill Bøjstrup",
                      "subject": "Machine Learning in Finance, Aarhus University"})
    pdf.set_toc(toc)
    LECTURES.mkdir(parents=True, exist_ok=True)
    target = LECTURES / f"session_{n:02d}.pdf"
    pdf.save(target, garbage=4, deflate=True, clean=True)
    print(f"  {target.relative_to(ROOT)}: {pdf.page_count} pages ({len(slides)} slides, "
          f"{sum(d.page_count for d in steps.values())} step pages), "
          f"{target.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s")
    return sha(html_path)


def build_site_pdf(browser, base_url, page_name, target):
    """A course page printed with its own print styles, as Ctrl+P would."""
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    page.goto(f"{base_url}/{page_name}", wait_until="networkidle")
    if page_name == "math.html":                       # KaTeX typesets after load
        page.wait_for_function("() => document.querySelectorAll('.katex').length > 100", timeout=60_000)
        page.wait_for_timeout(500)
    data = page.pdf(format="A4", print_background=True,
                    margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"})
    context.close()
    doc = pymupdf.open("pdf", data)
    doc.set_metadata({"title": page.title() if False else page_name, "author": "Jonas Theill Bøjstrup",
                      "subject": "Machine Learning in Finance, Aarhus University"})
    doc.save(target, garbage=4, deflate=True, clean=True)
    print(f"  {target.relative_to(ROOT)}: {doc.page_count} pages, {target.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="rebuild everything")
    ap.add_argument("--session", type=int, action="append", help="rebuild this session (repeatable)")
    args = ap.parse_args()

    decks = sorted(int(p.parent.name[-2:]) for p in ROOT.glob("session_*/session_*.html")
                   if re.fullmatch(r"session_\d\d\.html", p.name))
    stamps = json.loads(STAMP.read_text()) if STAMP.exists() else {}

    def stale(key, source, target):
        return args.all or stamps.get(key) != sha(source) or not target.exists()

    if args.session:
        deck_todo, site_todo = args.session, []
    else:
        deck_todo = [n for n in decks if stale(f"{n:02d}", ROOT / f"session_{n:02d}" / f"session_{n:02d}.html",
                                               LECTURES / f"session_{n:02d}.pdf")]
        site_todo = [k for k, (src, target) in SITE_PDFS.items() if stale(k, ROOT / src, target)]
    if not deck_todo and not site_todo:
        print("every PDF is up to date")
        return 0

    httpd = serve()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            for key in site_todo:
                src, target = SITE_PDFS[key]
                print(key)
                build_site_pdf(browser, base_url, src, target)
                stamps[key] = sha(ROOT / src)
                STAMP.parent.mkdir(parents=True, exist_ok=True)
                STAMP.write_text(json.dumps(stamps, indent=2, sort_keys=True) + "\n")
            for n in deck_todo:
                print(f"session {n}")
                stamps[f"{n:02d}"] = build_deck(browser, base_url, n)
                STAMP.write_text(json.dumps(stamps, indent=2, sort_keys=True) + "\n")
            browser.close()
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
