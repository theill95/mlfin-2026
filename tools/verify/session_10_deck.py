# -*- coding: utf-8 -*-
"""Run every {pyodide} cell of the Session 10 deck, in document order, in ONE
shared namespace - the same way the browser runtime does. Reports the printed
output and the displayed value of each cell so the slide text can be checked
against reality.

The two "Your turn" cells ship with `...` blanks: they may raise or print a
blank, and nothing later may depend on them. Every other cell must run clean,
or the script exits with an error.
"""
import ast
import io
import os
import re
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

os.chdir(str(Path(__file__).resolve().parents[2] / "session_10"))
import matplotlib
matplotlib.use("Agg")

QMD = "session_10.qmd"
src = open(QMD, encoding="utf-8").read()
body = src.split("\n---\n", 1)[-1]          # drop the YAML header

lines = body.split("\n")
cells = []                                  # (kind, title, options, code)
title = "(top)"
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("# "):
        title = re.sub(r"\[.*?\]\{.*?\}|\{.*?\}", "", ln[2:]).strip()
    m = re.match(r"^```\{(pyodide|python)\}\s*$", ln)
    if m:
        j = i + 1
        buf = []
        while j < len(lines) and not lines[j].startswith("```"):
            buf.append(lines[j])
            j += 1
        opts = [b for b in buf if b.startswith("#|")]
        code = "\n".join(b for b in buf if not b.startswith("#|"))
        cells.append((m.group(1), title, " ".join(opts), code))
        i = j
    i += 1

MAY_FAIL = ("Your turn",)

ns = {}
n_ok = n_err = 0
live = [c for c in cells if c[0] == "pyodide"]
print(f"{len(live)} live cells found\n" + "=" * 78)
for kind, title, opts, code in live:
    may_fail = any(k in title for k in MAY_FAIL)
    out = io.StringIO()
    disp = None
    err = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            tree = ast.parse(code)
            with redirect_stdout(out):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    exec(compile(ast.Module(tree.body[:-1], []), "<cell>", "exec"), ns)
                    disp = eval(compile(ast.Expression(tree.body[-1].value), "<cell>", "eval"), ns)
                else:
                    exec(compile(code, "<cell>", "exec"), ns)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    warn_txt = "; ".join(sorted({w.category.__name__ + ": " + str(w.message)[:70] for w in caught
                                 if not issubclass(w.category, (DeprecationWarning, FutureWarning))}))
    if err and not may_fail:
        status, n_err = "ERR ", n_err + 1
    else:
        status, n_ok = ("ok  (blank)" if err else "ok  "), n_ok + 1
    print(f"[{status}] {title}")
    if err:
        print(f"        {err}")
    o = out.getvalue().strip()
    if o:
        print("        OUT: " + o.replace("\n", "\n             ")[:900])
    if disp is not None and not hasattr(disp, "savefig"):
        rep = repr(disp)
        print("        => " + (rep[:500] + " ...").replace("\n", "\n           ") if len(rep) > 500
              else "        => " + rep.replace("\n", "\n           "))
    if warn_txt:
        print("        WARN: " + warn_txt[:300])
print("=" * 78)
print(f"{n_ok} ok, {n_err} unexpected errors")
sys.exit(1 if n_err else 0)
