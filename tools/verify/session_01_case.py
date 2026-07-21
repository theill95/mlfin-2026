from pathlib import Path
# -*- coding: utf-8 -*-
"""Cumulative check: load data, then exec every solution IN ORDER in ONE shared
namespace, proving the investigation chains correctly (later answers reuse the
values earlier ones stored)."""
import ast, io, os, re, json
from contextlib import redirect_stdout

os.chdir(str(Path(__file__).resolve().parents[2] / "session_01"))  # so the loader finds data/
NB = "session_01_case.ipynb"
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

# 1) run the provided loader cell
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "def load_closes" in src:
        run(src)
        print("loader:", "aapl_closes" in ns, "len", len(ns.get("aapl_closes", [])))
        break

# 2) exec each solution in order, shared namespace
qid = None
qre = re.compile(r"^### .*?(Q\d+)")
results, errors = [], []
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] != "markdown":
        continue
    m = qre.search(src.splitlines()[0]) if src.strip() else None
    if m:
        qid = m.group(1)
    mm = re.search(r"```python\n(.*?)```", src, re.S)
    if "✅ Solution" in src and mm:
        try:
            stdout, disp = run(mm.group(1))
            piece = []
            if stdout: piece.append("OUT[" + stdout.replace("\n", " / ") + "]")
            if disp is not None: piece.append("=> " + repr(disp))
            results.append((qid, "  ".join(piece)))
        except Exception as e:
            results.append((qid, f"!!! {type(e).__name__}: {e}"))
            errors.append(qid)

for qid, r in results:
    print(f"{str(qid):4} {r}")
print("=" * 60)
print(f"{len(results)} solutions, {len(errors)} errored")
print("final saved state:",
      {k: round(ns[k], 4) for k in ["aapl_return","ko_return","aapl_risk","ko_risk"] if k in ns})
