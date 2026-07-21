from pathlib import Path
# -*- coding: utf-8 -*-
"""Run every solution block from the Session 2 exercises the way a notebook
would (executing a trailing expression for display) and print its result, to
cross-check against the answer stated in each solution note. Each solution is
run in a FRESH namespace, which also proves every solution is self-contained."""
import ast, io, re, json
from contextlib import redirect_stdout

NB = str(Path(__file__).resolve().parents[2] / "session_02" / "session_02_exercises.ipynb")
nb = json.loads(open(NB, encoding="utf-8").read())

def run_like_notebook(src):
    tree = ast.parse(src)
    out = io.StringIO()
    disp = None
    with redirect_stdout(out):
        ns = {}
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<s>", "exec"), ns)
            disp = eval(compile(ast.Expression(tree.body[-1].value), "<s>", "eval"), ns)
        else:
            exec(src, ns)
    return out.getvalue().strip(), disp

sid_re = re.compile(r"^### (\w+) ")
current = None
results = []
for c in nb["cells"]:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if c["cell_type"] != "markdown":
        continue
    m = sid_re.match(src)
    if m:
        current = m.group(1)
    mm = re.search(r"```python\n(.*?)```", src, re.S)
    if "✅ Solution" in src and mm:
        try:
            stdout, disp = run_like_notebook(mm.group(1))
            pieces = []
            if stdout:
                pieces.append("OUT[" + stdout.replace("\n", " / ") + "]")
            if disp is not None:
                pieces.append("=> " + repr(disp))
            results.append((current, "  ".join(pieces) if pieces else "(no display)"))
        except Exception as e:
            results.append((current, f"!!! ERROR: {type(e).__name__}: {e}"))

bad = [r for r in results if "ERROR" in r[1]]
for sid, r in results:
    print(f"{str(sid):4} {r}")
print("\n" + "=" * 60)
print(f"{len(results)} solutions executed, {len(bad)} errored")
for sid, r in bad:
    print("  ", sid, r)
