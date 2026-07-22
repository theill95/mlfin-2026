from pathlib import Path
# -*- coding: utf-8 -*-
"""Cumulative check for the Session 4 case: run the quick load, then exec every
solution IN ORDER in ONE shared namespace, proving the investigation chains
correctly (later answers really do reuse what earlier ones stored)."""
import ast, io, json, os, re, warnings
from contextlib import redirect_stdout

os.chdir(str(Path(__file__).resolve().parents[2] / "session_04"))
import matplotlib
matplotlib.use("Agg")

NB = "session_04_case.ipynb"
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


for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "def data_path" in src:
        out, _ = run(src)
        print("QUICK LOAD:\n  " + out.replace("\n", "\n  "))
        break

qid = None
results, errors = [], []
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"])
    first = src.splitlines()[0] if src.strip() else ""
    m = re.match(r"^### (Q\d+) ", first)
    if m:
        qid = m.group(1)
    if "✅ Solution" not in src:
        continue
    mm = re.search(r"```python\n(.*?)```", src, re.S)
    if not mm:
        continue
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stdout, disp = run(mm.group(1))
        bad = [w for w in caught if "SettingWithCopy" in w.category.__name__]
        piece = []
        if stdout:
            piece.append("OUT[" + stdout.replace("\n", " / ")[:260] + "]")
        if disp is not None and not hasattr(disp, "savefig"):
            piece.append("=> " + repr(disp).replace("\n", " ")[:260])
        if bad:
            piece.append("!! SettingWithCopyWarning")
            errors.append(qid + " (warning)")
        results.append((qid, "  ".join(piece)))
    except Exception as e:
        results.append((qid, "!!! " + type(e).__name__ + ": " + str(e)))
        errors.append(qid)

print("\nSOLUTIONS:")
for qid, r in results:
    print(f"{str(qid):5} {r}")
print("=" * 78)
print(f"{len(results)} solutions, {len(errors)} problems", errors)
# Every variable the closing table promises must actually survive the chain.
PROMISED = ["target", "features", "table", "n", "p", "train", "test",
            "base_rmse", "base_mae", "pers_rmse", "pers_r2", "threshold",
            "precision", "recall", "spec"]
absent = [v for v in PROMISED if v not in ns]
print("closing table promises", len(PROMISED), "variables;",
      "all present" if not absent else f"MISSING: {absent}")
if absent:
    raise SystemExit(1)
