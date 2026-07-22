# -*- coding: utf-8 -*-
"""Build tools/colab_probe.ipynb: which formula layouts survive Colab?

Instructor-only. Not course material, not in the download bundle.

Round 1 established:

    A  markdown table + $...$     broken
    B  markdown table + \\(...\\)   literal \\text, never typeset
    C  HTML table + $...$         identical to A
    D  HTML table + \\(...\\)       identical to B
    E  HTML table + Unicode       works
    F  bullet list + $...$        works

So the table *type* is irrelevant: Colab will not typeset maths inside table
markup, and \\(...\\) is not a delimiter it knows anywhere. Round 2 asks the one
question that decides whether a two-column layout with real maths is possible
at all: does Colab typeset $...$ inside raw HTML that is *not* a table?

N is the diagnostic. If N fails, no HTML layout can carry LaTeX and the answer
is settled. If N works, G and H give the two-column look back.

    python tools/colab_probe.ipynb.py
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "colab_probe.ipynb"

ROWS = [
    ("Mean absolute error", r"\text{MAE}=\dfrac{1}{n}\sum_i \lvert y_i-\hat{y}_i \rvert"),
    ("Mean squared error", r"\text{MSE}=\dfrac{1}{n}\sum_i (y_i-\hat{y}_i)^2"),
    ("Root mean squared error", r"\text{RMSE}=\sqrt{\text{MSE}}"),
    ("R squared", r"R^2=1-\dfrac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}"),
    ("Standardised value", r"z_i=\dfrac{x_i-\bar{x}}{s}"),
]

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


md(
"# Round 2: can any HTML layout carry real mathematics?\n\n"
"Round 1 settled that the table **type** does not matter. Colab refuses to "
"typeset maths inside table markup, markdown or HTML alike, and it does not "
"know the `\\(...\\)` delimiters at all.\n\n"
"**Read N first.** It is the diagnostic, and it decides the rest:\n\n"
"- if the fraction in **N** renders as real mathematics, a two-column layout is "
"possible and **G** or **H** will give you the table look back\n"
"- if **N** shows raw `\\dfrac` text, then no HTML layout can carry LaTeX in "
"Colab, and the honest choice is between **E** and **F** from round 1\n"
)

md("---\n\n## N. The diagnostic: maths inside a plain `<div>` and `<span>`\n\n"
   "No table anywhere. If Colab typesets maths in raw HTML at all, it happens here.")
md('<div>Inside a div: $\\dfrac{a}{b}$ and $\\sum_i x_i^2$</div>\n'
   '<p>Inside a p: $R^2=1-\\dfrac{SS_{res}}{SS_{tot}}$</p>\n'
   '<span>Inside a span: $\\sqrt{\\text{MSE}}$</span>')

md("---\n\n## G. Two columns with CSS grid, no table markup\n\n"
   "This is the one that would give you back what you liked: aligned columns, "
   "real mathematics, and not a `<table>` in sight.")
md('<div style="display:grid;grid-template-columns:max-content 1fr;'
   'gap:0.45rem 1.5rem;align-items:baseline">\n' +
   "".join(f'<div><strong>{label}</strong></div><div>${tex}$</div>\n'
           for label, tex in ROWS) +
   "</div>")

md("---\n\n## H. Definition list\n\n"
   "Same idea, using the element HTML already has for term-and-definition. "
   "Indented rather than columnar, but tidy.")
md("<dl>\n" +
   "".join(f'<dt><strong>{label}</strong></dt><dd>${tex}$</dd>\n'
           for label, tex in ROWS) +
   "</dl>")

md("---\n\n## I. HTML table, but display maths `$$...$$`\n\n"
   "Round 1 only tried inline `$...$` in a table. This checks whether the "
   "display delimiters behave differently.")
md('<table>\n<tr><th align="left">what</th><th align="left">formula</th></tr>\n' +
   "".join(f'<tr><td>{label}</td><td>$${tex}$$</td></tr>\n' for label, tex in ROWS) +
   "</table>")

md("---\n\n## J. Markdown table, display maths `$$...$$`")
md("| what | formula |\n|:--|:--|\n" +
   "".join(f"| {label} | $${tex}$$ |\n" for label, tex in ROWS))

md("---\n\n**Tell me: does N render real mathematics, and which of G, H, I, J "
   "look right?**")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
for i, cell in enumerate(nb.cells):
    cell["id"] = f"c{i:04d}"

nbf.write(nb, str(OUT), version=4)
print(f"wrote {OUT.relative_to(ROOT)}: {len(nb.cells)} cells, round 2")
