from pathlib import Path
# -*- coding: utf-8 -*-
"""Run every folded solution of the Session 11 exercises against the real data.

The setup cell runs once. The exercises build on each other (A4 uses the `rss`
A3 wrote, D4 the `search` from D3, J2 to J5 the `vol_table` from J1), so every solution runs
IN ORDER in ONE shared namespace, exactly as a student working top to bottom
would run them.
"""
import ast, io, json, os, re, sys, time, warnings
from contextlib import redirect_stdout

os.chdir(str(Path(__file__).resolve().parents[2] / "session_11"))
import matplotlib
matplotlib.use("Agg")

NB = "session_11_exercises.ipynb"
nb = json.loads(open(NB, encoding="utf-8").read())

base = {}
setup_src = None
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "def data_path" in src:
        setup_src = src
        break
assert setup_src, "setup cell not found"
out = io.StringIO()
with redirect_stdout(out):
    exec(setup_src, base)
print("SETUP:", out.getvalue().strip().replace("\n", " | "))


def run(src, ns):
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


sid = None
section_id = None
ns = None
results, errors = [], []
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"])
    first = src.splitlines()[0] if src.strip() else ""
    m = re.match(r"^### ([A-Z]\d+) ", first)
    if m:
        sid = m.group(1)
    if "✅ Solution" not in src:
        continue
    mm = re.search(r"```python\n(.*?)```", src, re.S)
    if not mm:
        continue
    if ns is None:
        ns = dict(base)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            stdout, disp = run(mm.group(1), ns)
            took = time.perf_counter() - t0
        bad = [w for w in caught if "SettingWithCopy" in w.category.__name__]
        piece = [f"{took:5.1f}s"]
        if stdout:
            piece.append("OUT[" + stdout.replace("\n", " / ")[:220] + "]")
        if disp is not None:
            piece.append("=> " + repr(disp).replace("\n", " ")[:220])
        if bad:
            piece.append("!! SettingWithCopyWarning")
            errors.append(sid + " (warning)")
        results.append((sid, "  ".join(piece)))
    except Exception as e:
        results.append((sid, f"!!! {type(e).__name__}: {e}"))
        errors.append(sid)

for sid, r in results:
    print(f"{str(sid):4} {r}")
print("=" * 78)
print(f"{len(results)} solutions, {len(errors)} problems", errors)
