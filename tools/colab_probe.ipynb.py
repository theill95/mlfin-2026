# -*- coding: utf-8 -*-
"""Build tools/colab_probe.ipynb: which formula layouts survive Colab?

Instructor-only. It is not course material, it is not in the download bundle,
and students never see it.

Colab collapses a markdown table row that contains LaTeX, while Jupyter renders
it correctly. That much is established. What is *not* established is which of
the alternatives Colab is happy with, and guessing costs a push and a round
trip each time. So this notebook renders the same Session 4 formula card six
ways and lets one look settle it.

    python tools/colab_probe.ipynb.py

Then open the notebook in Colab, see which blocks look right, and tell the
generator to use that layout.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "colab_probe.ipynb"

# The Session 4 card: the hardest case, because two rows mix prose and maths.
ROWS = [
    ("Mean absolute error", r"\text{MAE}=\dfrac{1}{n}\sum_i \lvert y_i-\hat{y}_i \rvert",
     "MAE = (1/n) Σᵢ |yᵢ − ŷᵢ|"),
    ("Mean squared error", r"\text{MSE}=\dfrac{1}{n}\sum_i (y_i-\hat{y}_i)^2",
     "MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²"),
    ("Root mean squared error", r"\text{RMSE}=\sqrt{\text{MSE}}",
     "RMSE = √MSE"),
    ("R squared", r"R^2=1-\dfrac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}",
     "R² = 1 − Σᵢ(yᵢ − ŷᵢ)² / Σᵢ(yᵢ − ȳ)²"),
    ("Standardised value", r"z_i=\dfrac{x_i-\bar{x}}{s}",
     "zᵢ = (xᵢ − x̄) / s"),
]

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


md(
"# Which formula layout survives Colab?\n\n"
"Every block below is the **same five formulas**. Scroll through and note which "
"ones look like a proper two-column table with real mathematics.\n\n"
"Blocks are labelled **A** to **F**. Tell me the letters that render correctly.\n\n"
"Block A is the known-broken one, so it is the control: if A looks fine here, "
"then something else is going on and the rest of this notebook is moot."
)

md("---\n\n## A. Markdown table, `$...$`  (the broken control)")
md("| what | formula |\n|:--|:--|\n" +
   "".join(f"| {label} | ${tex}$ |\n" for label, tex, _ in ROWS))

md("---\n\n## B. Markdown table, `\\(...\\)` delimiters\n\n"
   "Same table, but the maths is delimited with `\\(` and `\\)` instead of `$`. "
   "If Colab's breakage comes from pairing `$` signs across rows, this fixes it "
   "while keeping the table.")
md("| what | formula |\n|:--|:--|\n" +
   "".join(f"| {label} | \\({tex}\\) |\n" for label, tex, _ in ROWS))

md("---\n\n## C. HTML table, `$...$`\n\n"
   "A real `<table>`, so the markdown table parser is never involved. The maths "
   "is still LaTeX, typeset in the page after the HTML is laid out.")
md('<table>\n<tr><th align="left">what</th><th align="left">formula</th></tr>\n' +
   "".join(f'<tr><td>{label}</td><td>${tex}$</td></tr>\n' for label, tex, _ in ROWS) +
   "</table>")

md("---\n\n## D. HTML table, `\\(...\\)` delimiters")
md('<table>\n<tr><th align="left">what</th><th align="left">formula</th></tr>\n' +
   "".join(f'<tr><td>{label}</td><td>\\({tex}\\)</td></tr>\n' for label, tex, _ in ROWS) +
   "</table>")

md("---\n\n## E. HTML table, plain Unicode, no LaTeX at all\n\n"
   "Cannot fail, because nothing has to typeset it. The question is only whether "
   "you find it good enough to read.")
md('<table>\n<tr><th align="left">what</th><th align="left">formula</th></tr>\n' +
   "".join(f'<tr><td>{label}</td><td><code>{uni}</code></td></tr>\n'
           for label, _, uni in ROWS) +
   "</table>")

md("---\n\n## F. Bullet list  (what is published right now)")
md("".join(f"- **{label}**: ${tex}$\n" for label, tex, _ in ROWS))

md("---\n\nWhich letters rendered as a proper table with proper mathematics?")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
for i, cell in enumerate(nb.cells):
    cell["id"] = f"c{i:04d}"

nbf.write(nb, str(OUT), version=4)
print(f"wrote {OUT.relative_to(ROOT)}: {len(nb.cells)} cells, 6 layouts")
