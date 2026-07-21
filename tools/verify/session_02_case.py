from pathlib import Path
# -*- coding: utf-8 -*-
"""Cumulative check for the Session 2 case: run the quick-load cell, then exec
every solution IN ORDER in ONE shared namespace, proving the investigation
chains correctly (later answers reuse what earlier ones stored)."""
import ast, io, os, re, json
from contextlib import redirect_stdout

os.chdir(str(Path(__file__).resolve().parents[2] / "session_02"))  # so the loader finds data/
NB = "session_02_case.ipynb"
nb = json.loads(open(NB, encoding="utf-8").read())
ns = {}

def run(src):
    tree = ast.parse(src)
    out = io.StringIO()
    disp = None
    with redirect_stdout(out):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            exec(compile(ast.Module(tree.body[:-1], []), "<s>", "exec"), ns)
            disp = eval(compile(ast.Expression(tree.body[-1].value), "<s>", "eval"), ns)
        else:
            exec(src, ns)
    return out.getvalue().strip(), disp

# 1) the provided quick-load cell
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "def load_closes" in src:
        out, _ = run(src)
        print("QUICK LOAD:")
        print("  " + out.replace("\n", "\n  "))
        break

# 2) every solution, in order, shared namespace
qid = None
qre = re.compile(r"^### .*?(Q\d+)")
results, errors = [], []
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] != "markdown":
        continue
    first = src.splitlines()[0] if src.strip() else ""
    m = qre.search(first)
    if m:
        qid = m.group(1)
    mm = re.search(r"```python\n(.*?)```", src, re.S)
    if "✅ Solution" in src and mm:
        try:
            stdout, disp = run(mm.group(1))
            piece = []
            if stdout:
                piece.append("OUT[" + stdout.replace("\n", " / ") + "]")
            if disp is not None:
                piece.append("=> " + repr(disp))
            results.append((qid, "  ".join(piece)))
        except Exception as e:
            results.append((qid, f"!!! {type(e).__name__}: {e}"))
            errors.append(qid)

print("\nSOLUTIONS:")
for qid, r in results:
    print(f"{str(qid):4} {r}")
print("=" * 70)
print(f"{len(results)} solutions, {len(errors)} errored")
saved = ["aapl_vol", "ko_vol", "riskiest", "calmest"]
print("final saved state:",
      {k: (round(ns[k], 5) if isinstance(ns.get(k), float) else ns.get(k))
       for k in saved if k in ns})
print("vol dict:", {k: round(v, 5) for k, v in ns.get("vol", {}).items()})
print("functions defined:", [f for f in ["daily_returns", "mean", "volatility"] if f in ns])
