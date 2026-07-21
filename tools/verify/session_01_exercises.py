from pathlib import Path
# -*- coding: utf-8 -*-
"""Execute every solution code block the way a notebook would (running the
final expression for display) and print its result, to cross-check against the
answer stated in each solution note."""
import ast, io, re, json
from contextlib import redirect_stdout

NB = str(Path(__file__).resolve().parents[2] / "session_01" / "session_01_exercises.ipynb")
nb = json.loads(open(NB, encoding="utf-8").read())

def run_like_notebook(src):
    """exec all but a trailing expression; eval that for the 'displayed' value."""
    tree = ast.parse(src)
    out = io.StringIO()
    disp = None
    with redirect_stdout(out):
        ns = {}
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<s>", "exec"), ns)
            val = eval(compile(ast.Expression(tree.body[-1].value), "<s>", "eval"), ns)
            disp = val  # notebook only displays a non-None trailing expression
        else:
            exec(src, ns)
    return out.getvalue().strip(), disp

cells = nb["cells"]
sid_re = re.compile(r"^### (\w+) ")
current = None
results = []
for c in cells:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if c["cell_type"] == "markdown":
        m = sid_re.match(src)
        if m:
            current = m.group(1)
        mm = re.search(r"```python\n(.*?)```", src, re.S)
        if "✅ Solution" in src and mm:
            code = mm.group(1)
            try:
                stdout, disp = run_like_notebook(code)
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
