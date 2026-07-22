# -*- coding: utf-8 -*-
"""Keep the markdown cells renderable by Google Colab, not just by Jupyter.

Colab does not use Jupyter's markdown renderer, and the two disagree in one
place that bites us: once a cell has emitted a block-level raw HTML element,
Colab stops treating what follows as markdown. Jupyter carries on parsing it,
so the cell looks perfect locally and arrives in Colab as a wall of literal
pipe characters, stacked one line per row.

That is exactly what happened to the toolkit card: a run of <p> chips followed
by the "Formulas you will reach for" table.

The rule this enforces is simple and costs nothing to obey:

    markdown first, raw HTML last, never markdown again after the HTML.

    python tools/verify/colab_markup.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted(ROOT.glob("session_*/session_*.ipynb"))

# Block-level HTML we emit on purpose. <details>/<summary> are the folded hints
# and solutions, which hold markdown by design and are handled by Colab fine,
# so they do not start a "no markdown after this" region.
BLOCK_HTML = re.compile(r"<(p|div|table|ul|ol|blockquote)\b", re.I)

# Markdown that silently stops working once raw HTML has opened.
MD_TABLE = re.compile(r"^\s*\|.*\|\s*$")
MD_HEADING = re.compile(r"^#{1,6}\s")

# Inline maths inside a table cell. Colab typesets $$...$$ in a table happily
# but collapses the row on $...$, while Jupyter renders both, which is what
# makes this so easy to ship unnoticed. Measured in Colab, not guessed: see
# "Maths in Colab" in tools/README.md. Maths outside tables is unaffected, so
# the rule is deliberately narrow.
DISPLAY_MATH = re.compile(r"\$\$.+?\$\$")
TABLE_ROW = re.compile(r"^\s*\|")

failures = []
checked = 0

for path in NOTEBOOKS:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        checked += 1
        lines = "".join(cell["source"]).split("\n")

        for n, line in enumerate(lines):
            if not TABLE_ROW.match(line):
                continue
            # Take the display maths out, then any $ still standing is inline.
            if "$" in DISPLAY_MATH.sub("", line):
                failures.append(
                    f"{path.name} cell {index}: inline $...$ maths in a table "
                    f"cell on line {n + 1}. Colab collapses the row. Use $$...$$ "
                    f"instead:\n      {line.strip()[:78]}"
                )

        html_at = None
        for n, line in enumerate(lines):
            if html_at is None and BLOCK_HTML.search(line):
                html_at = n
                continue
            if html_at is None:
                continue
            what = ("a markdown table" if MD_TABLE.match(line)
                    else "a markdown heading" if MD_HEADING.match(line)
                    else None)
            if what:
                head = next((l for l in lines if l.strip()), "")[:44]
                failures.append(
                    f"{path.name} cell {index} [{head}]: {what} on line {n + 1} "
                    f"follows raw HTML opened on line {html_at + 1}. Colab will "
                    f"render it literally. Move it into its own cell."
                )
                break

print(f"checked {checked} markdown cells across {len(NOTEBOOKS)} notebooks")
if failures:
    print(f"\n{len(failures)} cell(s) Colab would render differently from Jupyter:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("no markdown is stranded after raw HTML: Colab and Jupyter will agree")
