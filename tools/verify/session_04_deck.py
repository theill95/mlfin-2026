from pathlib import Path
# -*- coding: utf-8 -*-
"""Run every {pyodide} cell of the Session 4 deck, in document order, in ONE
shared namespace - the same way the browser runtime does. Reports the displayed
value of each cell so the slide text can be checked against reality.

Cells whose title contains an expected-error marker are checked to actually
raise. Everything else must run clean.
"""
import ast, io, re, sys, os, warnings
from contextlib import redirect_stdout

os.chdir(str(Path(__file__).resolve().parents[2] / "session_04"))
import matplotlib
matplotlib.use("Agg")

QMD = "session_04.qmd"
src = open(QMD, encoding="utf-8").read()

# strip the yaml header
body = src.split("\n---\n", 2)[-1]

# slide titles, to label each cell
lines = body.split("\n")
cells = []           # (title, options, code)
title = "(top)"
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("# "):
        title = re.sub(r"\{.*?\}|\[.*?\]\{.*?\}", "", ln[2:]).strip()
    m = re.match(r"^```\{(pyodide|python)\}\s*$", ln)
    if m:
        kind = m.group(1)
        j = i + 1
        buf = []
        while j < len(lines) and not lines[j].startswith("```"):
            buf.append(lines[j])
            j += 1
        opts = [b for b in buf if b.startswith("#|")]
        code = "\n".join(b for b in buf if not b.startswith("#|"))
        cells.append((kind, title, " ".join(opts), code))
        i = j
    i += 1

# No deliberate-error slide in this deck: every live cell must run clean.
EXPECT_ERROR = ()

ns = {}
n_ok = n_err = 0
print(f"{len(cells)} code cells found\n" + "=" * 78)
for kind, title, opts, code in cells:
    if kind != "pyodide":
        continue
    expect_err = any(k in title for k in EXPECT_ERROR)
    out = io.StringIO()
    disp = None
    err = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tree = ast.parse(code)
            with redirect_stdout(out):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    exec(compile(ast.Module(tree.body[:-1], []), "<c>", "exec"), ns)
                    disp = eval(compile(ast.Expression(tree.body[-1].value), "<c>", "eval"), ns)
                else:
                    exec(code, ns)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    warn_txt = "; ".join(sorted({str(w.category.__name__) + ": " + str(w.message)[:70]
                                 for w in caught})) if 'caught' in dir() else ""
    status = "ERR " if err else "ok  "
    if expect_err:
        status = "ok  (expected error)" if err else "!!! SHOULD HAVE ERRORED"
    if err and not expect_err:
        n_err += 1
    else:
        n_ok += 1
    print(f"[{status}] {title}")
    if err:
        print(f"        {err}")
    o = out.getvalue().strip()
    if o:
        print("        OUT: " + o.replace("\n", "\n             ")[:600])
    if disp is not None and not hasattr(disp, "savefig"):
        rep = repr(disp)
        if len(rep) > 400:
            rep = rep[:400] + " ..."
        print("        => " + rep.replace("\n", "\n           "))
    if warn_txt:
        print("        WARN: " + warn_txt[:300])
print("=" * 78)
print(f"{n_ok} ok, {n_err} unexpected errors")
