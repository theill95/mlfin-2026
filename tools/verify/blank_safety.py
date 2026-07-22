# -*- coding: utf-8 -*-
"""Prove that pressing Run All on an untouched notebook is safe.

Every task cell ships with `...` placeholders. A student who runs the whole
notebook before filling anything in must see no errors, or the notebook looks
broken on first contact. Two things can go wrong, and only the first one is
obvious:

  raises   `x = ...` then `x.mean()`  ->  AttributeError on 'ellipsis'
  HANGS    `while ...:`               ->  Ellipsis is truthy, so it never stops

The second is much worse than a traceback and does not show up in a quick look,
which is exactly why this runs on every release.

    python tools/verify/blank_safety.py

This is a static check, so it is fast enough to run every time, unlike executing
eight notebooks. It complements `nbconvert --execute` rather than replacing it.
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted(ROOT.glob("session_*/session_*_exercises.ipynb")) + \
            sorted(ROOT.glob("session_*/session_*_case.ipynb"))

# Builtins and library calls that evaluate their argument immediately.
EAGER = {"len", "sum", "sorted", "range", "min", "max", "abs", "round", "int",
         "float", "str", "list", "dict", "tuple", "set", "enumerate", "zip"}


def is_ellipsis(node):
    return isinstance(node, ast.Constant) and node.value is Ellipsis


class Blanks(ast.NodeVisitor):
    """Find placeholders that are used rather than merely assigned."""

    def __init__(self):
        self.blank = set()      # names currently bound to `...`
        self.problems = []

    def visit_Assign(self, node):
        # `x = ...` binds a placeholder; `x = something_real` clears it
        self.visit(node.value)
        for target in node.targets:
            names = ([target] if isinstance(target, ast.Name)
                     else [e for e in getattr(target, "elts", [])
                           if isinstance(e, ast.Name)])
            if is_ellipsis(node.value):
                for n in names:
                    self.blank.add(n.id)
            elif isinstance(node.value, ast.Tuple) and \
                    all(is_ellipsis(e) for e in node.value.elts):
                for n in names:
                    self.blank.add(n.id)
            else:
                for n in names:
                    self.blank.discard(n.id)

    def _flag(self, what):
        self.problems.append(what)

    def visit_While(self, node):
        if any(is_ellipsis(n) for n in ast.walk(node.test)):
            self._flag("`while ...` never terminates: Ellipsis is truthy")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.blank:
            self._flag(f"`{node.value.id}.{node.attr}` on a placeholder")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.blank:
            self._flag(f"indexing into the placeholder `{node.value.id}`")
        self.generic_visit(node)

    def visit_For(self, node):
        it = node.iter
        if is_ellipsis(it) or (isinstance(it, ast.Name) and it.id in self.blank):
            self._flag("iterating a placeholder")
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in EAGER:
                for a in node.args:
                    if is_ellipsis(a) or (isinstance(a, ast.Name) and a.id in self.blank):
                        self._flag(f"`{func.id}(...)` on a placeholder")
            if func.id in self.blank:
                self._flag(f"calling the placeholder `{func.id}`")
        elif isinstance(func, ast.Attribute):
            # pd.DataFrame({'a': ..., 'b': ...}) raises "all scalar values".
            # With at least one real column pandas broadcasts the placeholder
            # over that column's index instead, which is harmless, so only an
            # entirely blank literal is a problem.
            owner = getattr(func.value, "id", "")
            if owner in {"pd", "np"}:
                for a in node.args:
                    if is_ellipsis(a):
                        self._flag(f"`{owner}.{func.attr}(...)` on a bare placeholder")
                    elif isinstance(a, ast.Dict) and a.values and \
                            all(is_ellipsis(v) for v in a.values):
                        self._flag(f"`{owner}.{func.attr}` built only from placeholders")
                    elif isinstance(a, (ast.List, ast.Tuple)) and a.elts and \
                            all(is_ellipsis(e) for e in a.elts):
                        self._flag(f"`{owner}.{func.attr}` built only from placeholders")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        for side in (node.left, node.right):
            if isinstance(side, ast.Name) and side.id in self.blank:
                self._flag(f"arithmetic with the placeholder `{side.id}`")
        self.generic_visit(node)


failures = []
checked = 0

for path in NOTEBOOKS:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "..." not in src:
            continue
        # a cell that is meant to raise is exempt
        if "raises-exception" in cell.get("metadata", {}).get("tags", []):
            continue
        checked += 1
        try:
            tree = ast.parse(src)
        except SyntaxError:
            failures.append(f"{path.name}: a work cell does not parse")
            continue
        finder = Blanks()
        finder.visit(tree)
        if finder.problems:
            label = "?"
            for j in range(i - 1, max(-1, i - 4), -1):
                head = "".join(cells[j]["source"]).split("\n")[0]
                if head.startswith("###"):
                    label = re.sub(r"\s+", " ", head)[4:44]
                    break
            for p in dict.fromkeys(finder.problems):
                failures.append(f"{path.name} [{label}] {p}")

print(f"checked {checked} work cells across {len(NOTEBOOKS)} notebooks")
if failures:
    print(f"\n{len(failures)} blank-safety problem(s):")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("every work cell is safe to run with its placeholders untouched")
