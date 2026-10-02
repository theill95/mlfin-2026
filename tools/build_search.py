"""Build assets/search-index.json: everything the search page searches.

One entry per lecture slide, exercise and case question, cheatsheet row,
glossary term, and section of the maths page and of the other pages, each
with the link that opens it. It reads the rendered decks, the notebooks and
the built pages, so run it after those (release.py does).

    python tools/build_search.py
"""
import base64
import html
import json
import keyword
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "search-index.json"
COLAB = ("https://colab.research.google.com/github/theill95/mlfin-2026/blob/main/"
         "session_{n:02d}/session_{n:02d}_{kind}.ipynb")
NOTEBOOK_SESSIONS = (1, 2, 3, 4, 5, 6, 8, 9, 11, 12)

# Pages searched section by section: file, the label results carry.
PAGES = [
    ("index.html", "Sessions"),
    ("case.html", "The case"),
    ("setup.html", "Setup & tools"),
    ("resources.html", "Resources"),
    ("downloads.html", "Downloads"),
    ("exam_info.html", "Exam info"),
]

DROP = re.compile(r"<(script|style|svg|aside|noscript|template)\b.*?</\1\s*>", re.S | re.I)
MATH_INLINE = re.compile(r'<span class="math inline">(.*?)</span>', re.S)
MATH_DISPLAY = re.compile(r'<span class="math display">.*?</span>', re.S)
TEX_DISPLAY = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.S)
TEX_INLINE = re.compile(r"\\\((.*?)\\\)", re.S)
# What a TeX command reads as, in a line of plain text.
TEX_WORDS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Lambda": "Λ", "Omega": "Ω",
    "Phi": "Φ", "Pi": "Π", "sum": "Σ", "prod": "Π", "times": "×", "cdot": "·", "le": "≤",
    "leq": "≤", "ge": "≥", "geq": "≥", "approx": "≈", "ne": "≠", "neq": "≠", "in": "∈",
    "to": "→", "rightarrow": "→", "infty": "∞", "pm": "±", "ldots": "…", "dots": "…",
    "cdots": "…", "lvert": "|", "rvert": "|", "mid": "|", "min": "min", "max": "max",
    "log": "log", "exp": "exp", "top": "ᵀ",
}


def tex_text(tex):
    """A short formula as plain text: \\hat{y}_i \\le 0.5 reads y_i ≤ 0.5."""
    t = tex
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\(?:text|mathrm|mathbf|mathit|boldsymbol|operatorname|textbf|hat|bar|tilde)\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\%", "%").replace("\\{", "{").replace("\\}", "}")
    t = re.sub(r"\\([A-Za-z]+)", lambda m: TEX_WORDS.get(m.group(1), " "), t)
    t = re.sub(r"\\.", " ", t).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def tex_inline(m):
    """Inline maths inside HTML, as text the tag-stripping cannot mistake for a tag."""
    return html.escape(tex_text(html.unescape(m.group(1))), quote=False)
# A block ends a run of words; an inline tag sits inside one.
BLOCK = re.compile(r"</?(?:p|div|li|ul|ol|h[1-6]|section|table|tr|td|th|thead|tbody|br|pre|"
                   r"blockquote|dl|dt|dd|figure|figcaption|details|summary|header|footer|main)\b[^>]*>", re.I)
TAG = re.compile(r"<[^>]+>")
CHIP = re.compile(r'<span class="chip[^"]*">.*?</span>', re.S)
# Code words too common to say anything about a slide.
COMMON = set(keyword.kwlist) | {
    "print", "round", "len", "range", "self", "def", "import", "from", "True", "False", "None"}


def text_of(fragment):
    """The words a reader sees in a piece of HTML, on one line."""
    s = DROP.sub(" ", fragment)
    s = MATH_INLINE.sub(tex_inline, MATH_DISPLAY.sub(" ", s))
    s = TAG.sub("", BLOCK.sub(" ", s))
    s = TEX_DISPLAY.sub(" ", html.unescape(s))
    s = TEX_INLINE.sub(lambda m: tex_text(m.group(1)), s)
    return re.sub(r"\s+", " ", s).strip()


def clip(s, n):
    """At most n characters, cut at a space."""
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + " …"


def code_words(code):
    """The names a code cell uses, once each: how a search for a function finds it."""
    seen = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code):
        if len(w) > 2 and w not in COMMON and w not in seen:
            seen.append(w)
    return " ".join(seen)


def item(kind, label, title, url, text="", code=""):
    d = {"k": kind, "l": label, "t": title, "u": url}
    if text:
        d["x"] = text
    if code:
        d["c"] = code
    return d


# --------------------------------------------------------------- lectures
def lecture_items(n):
    path = f"session_{n:02d}/session_{n:02d}.html"
    page = (ROOT / path).read_text(encoding="utf-8")
    deck = html.unescape(re.search(r"<title>(.*?)</title>", page, re.S).group(1)).strip()
    label = f"Session {n} · lecture"
    out = []
    for sid, cls, body in re.findall(
            r'<section id="([^"]+)" class="([^"]*)"[^>]*>(.*?)</section>', page, re.S):
        assert "<section" not in body, f"{path}: a slide inside a slide, at {sid}"
        if sid == "title-slide":
            sub = re.search(r'<p class="subtitle"[^>]*>(.*?)</p>', body, re.S)
            out.append(item("lecture", label, f"Session {n} · {deck}", path,
                            text_of(sub.group(1)) if sub else ""))
            continue
        head = re.search(r"<h([12])[^>]*>(.*?)</h\1>", body, re.S)
        if not head:
            continue
        title = text_of(CHIP.sub("", head.group(2)))
        words = []
        for blob in re.findall(r'<script type="pyodide-\d+-contents">\s*(.*?)\s*</script>', body, re.S):
            cell = json.loads(base64.b64decode(blob))
            if cell["attr"].get("include", True) and cell["attr"].get("echo", True):
                words.append(cell["code"])
        prose = text_of(body[head.end():])
        out.append(item("lecture", label, title, f"{path}#/{sid}",
                        clip(prose, 700), code_words("\n".join(words))))
    return out


# ---------------------------------------------------- exercises and cases
def markdown_text(md):
    """Markdown as plain words: no code fences, emphasis, links or HTML."""
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    md = re.sub(r"<[^>]+>", " ", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = re.sub(r"\*\*|`|^\s*[#>]+\s*", "", md, flags=re.M)
    md = re.sub(r"(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])", "", md)
    md = TEX_DISPLAY.sub(" ", md)
    md = re.sub(r"\$([^$\n]+)\$", lambda m: tex_text(m.group(1)), md)
    return re.sub(r"\s+", " ", md).strip()


def notebook_items(n, kind):
    nb = json.loads((ROOT / f"session_{n:02d}/session_{n:02d}_{kind}.ipynb").read_text(encoding="utf-8"))
    url = COLAB.format(n=n, kind=kind)
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        m = re.match(r"\s*###\s+((?:[A-Z]\d+|Q\d+)\s+·\s+.+?)\s*(★[★☆]*)?\s*(·\s*revisits S\d+)?\s*$",
                     src.splitlines()[0] if src.strip() else "")
        if not m:
            continue
        title = m.group(1).strip()
        stars = m.group(2) or ""
        body = markdown_text("\n".join(src.splitlines()[1:]))
        if kind == "exercises":
            label = f"Session {n} · exercises" + (f" · {stars}" if stars else "")
            out.append(item("exercise", label, title, url, clip(body, 300)))
        else:
            out.append(item("case", f"Session {n} · the case, Part {n}", title, url, clip(body, 300)))
    return out


# ------------------------------------------------------------- cheatsheet
def cheatsheet_items():
    page = (ROOT / "cheatsheet.html").read_text(encoding="utf-8")
    out = []
    for sec_id, sec_body in re.findall(r'<section class="cs-section" id="([^"]+)">(.*?)</section>', page, re.S):
        sec_title = text_of(re.search(r"<h2>(.*?)</h2>", sec_body, re.S).group(1))
        for since, row_id, sig, desc in re.findall(
                r'<div class="cs-row" data-since="(\d+)">\s*<div class="cs-sig" id="([^"]+)">(.*?)</div>\s*'
                r'<div class="cs-desc">(.*?)<span class="cs-since">', sec_body, re.S):
            out.append(item("cheatsheet", f"Cheatsheet · {sec_title} · Session {since}",
                            text_of(sig), f"cheatsheet.html#{row_id}", text_of(desc)))
    assert out, "no cheatsheet rows found: has the row markup changed?"
    return out


# --------------------------------------------------------------- glossary
def glossary_items():
    page = (ROOT / "revision.html").read_text(encoding="utf-8")
    out = []
    for gid, term, sessions, defn in re.findall(
            r'<dt id="(g-[^"]+)">(.*?)<span class="g-s">(.*?)</span></dt>\s*<dd>(.*?)</dd>', page, re.S):
        out.append(item("glossary", f"Glossary · {text_of(sessions)}", text_of(term),
                        f"revision.html#{gid}", text_of(defn)))
    assert len(out) > 40, "the glossary was not found on the revision page"
    return out


# ------------------------------------------------------------------ maths
def maths_items():
    page = (ROOT / "math.html").read_text(encoding="utf-8")
    main = page[page.index('<main'):page.index("</main>")]
    out = []
    parts = re.split(r'(<h2 id="[^"]+">.*?</h2>)', main, flags=re.S)
    for head, body in zip(parts[1::2], parts[2::2]):
        hid = re.search(r'id="([^"]+)"', head).group(1)
        htitle = text_of(head)
        url = f"math.html#{hid}"
        subs = re.split(r"(<h3[^>]*>.*?</h3>)", body, flags=re.S)
        out.append(item("maths", "Maths brush-up", htitle, url, clip(text_of(subs[0]), 400)))
        for sub_head, sub_body in zip(subs[1::2], subs[2::2]):
            out.append(item("maths", f"Maths brush-up · {htitle}", text_of(sub_head), url,
                            clip(text_of(sub_body), 400)))
    return out


# ------------------------------------------------------------------ pages
def page_items(name, label):
    page = (ROOT / name).read_text(encoding="utf-8")
    main = page[page.index("<main"):page.index("</main>")]
    main = re.sub(r"<footer>.*?</footer>", " ", main, flags=re.S)
    out = []
    if name == "index.html":
        for q, a in re.findall(r'<details class="faq">\s*<summary>(.*?)</summary>(.*?)</details>', main, re.S):
            out.append(item("page", "Sessions · Questions students ask", text_of(q), "index.html#faq", text_of(a)))
        main = re.sub(r'<section class="card" id="faq">.*?</section>', " ", main, flags=re.S)
    heads = list(re.finditer(r"<h2([^>]*)>(.*?)</h2>", main, re.S))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(main)
        own = re.search(r'id="([^"]+)"', h.group(1))
        # A heading without an id takes the id of the card it opens.
        card = re.search(r'<(?:section|div)[^>]*id="([^"]+)"[^>]*>\s*(?:<details[^>]*>\s*<summary>\s*)?$',
                         main[max(0, h.start() - 300):h.start()])
        anchor = own.group(1) if own else card.group(1) if card else ""
        url = name + (f"#{anchor}" if anchor else "")
        title = text_of(re.sub(r'<span class="tag[^"]*">.*?</span>', "", h.group(2)))
        out.append(item("page", label, title, url, clip(text_of(main[h.end():end]), 400)))
    return out


def main():
    items = glossary_items() + cheatsheet_items() + maths_items()
    for n in range(1, 14):
        items += lecture_items(n)
    for n in NOTEBOOK_SESSIONS:
        items += notebook_items(n, "case")
    for n in NOTEBOOK_SESSIONS:
        items += notebook_items(n, "exercises")
    for name, label in PAGES:
        items += page_items(name, label)
    OUT.write_text(json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    counts = {}
    for d in items:
        counts[d["k"]] = counts.get(d["k"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)}: {len(items)} entries, {OUT.stat().st_size // 1024} KB",
          ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
