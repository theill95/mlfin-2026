"""Check that an earlier lecture cell can be rerun after a later topic reuses names."""

import contextlib
import io
import os
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SESSION = Path(__file__).resolve().parents[2] / "session_09"
os.chdir(SESSION)
source = (SESSION / "session_09.qmd").read_text(encoding="utf-8")
body = source.split("\n---\n", 1)[-1]

cells = []
for match in re.finditer(r"(?m)^```\{pyodide\}\s*\n(.*?)^```", body, re.S | re.M):
    lines = match.group(1).splitlines()
    code = "\n".join(line for line in lines if not line.startswith("#|"))
    cells.append(("#| autorun: false" not in lines, code))

previous = {}
credit_cell = None
credit_state = None
market_state = None
for autorun, code in cells:
    current = previous.copy()  # the browser snapshots earlier names for each cell
    if autorun:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            exec(compile(code, "<lecture cell>", "exec"), current)
    if 'precision_score(test["default"], predicted)' in code:
        credit_cell, credit_state = code, current
    if "test" in current and "default" not in current["test"].columns:
        market_state = current
        break
    previous = current

assert credit_cell and credit_state and market_state
assert "default" in credit_state["test"].columns
assert "default" not in market_state["test"].columns
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(credit_cell, "<revisited credit cell>", "exec"), credit_state)
assert output.getvalue().splitlines() == ["0.678", "0.311"]
print("Session 9 credit cell retains its data after the market section runs")
