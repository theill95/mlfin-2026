# -*- coding: utf-8 -*-
"""Generate setup_check.ipynb: the notebook students run on their exam laptop.

It checks, with no internet, that the laptop has a supported Python, the
package versions in requirements.txt, every data file of the course, and that
one model of each kind the course used fits and a figure draws. It ends with one
verdict. It never downloads anything: the point is to prove the laptop works
offline, so a missing file is reported, not fetched.

    python tools/generators/setup_check.py

Writes setup_check.ipynb at the top of the repository; the download bundle puts
it at the top of the course ZIP, next to data/, and the Exam Info page links it.
The versions are read from requirements.txt when the notebook is generated.
"""
import re
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "setup_check.ipynb"

PINS = {}
for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    m = re.fullmatch(r"([A-Za-z0-9_-]+)\s*(~=|==)\s*([\d.]+)", line)
    if m:
        name, op, version = m.groups()
        # ~=2.3.0 means any 2.3.x; ==2.1.4 means exactly that
        PINS[name] = version.rsplit(".", 1)[0] + "." if op == "~=" else version

# The course's data files and their row counts (header excluded).
DATA = {
    "prices.csv": 27676,
    "market_features.csv": 2376,
    "credit.csv": 30000,
    "aarhus_houses.csv": 7621,
    "power.csv": 26133,
    "aapl_2024_closes.csv": 252,
}

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md("""# Setup check

Run this notebook **once on the laptop you will bring to the exam, with Wi-Fi switched
off**. It checks four things: your Python, the course's packages, the data files, and
that every kind of model from the course fits and a figure draws. If the last cell
says **ready**, your laptop can do everything the course did, offline.

How to run it:

1. Unzip the course ZIP from the Downloads page, if you have not already.
2. In VS Code, **File → Open Folder** and pick the unzipped folder (the one with
   `data/` and this notebook in it).
3. Open `setup_check.ipynb`, choose your Python 3.14 when VS Code asks for a kernel,
   and press **Run All**.

Nothing here downloads anything. A missing file or package is reported, not fetched.""")

code('''import sys
problems = []

print("Python", sys.version.split()[0])
if not ((3, 12) <= sys.version_info[:2] <= (3, 14)):
    problems.append("Python should be 3.12, 3.13 or 3.14 (3.14 recommended)")''')

md("## 1 · The packages\n\nEach version is compared with the one the course was checked with.")

pins_literal = "{\n" + "".join(f'    "{k}": "{v}",\n' for k, v in PINS.items()) + "}"
code(f'''from importlib import metadata

WANTED = {pins_literal}
# A version ending in a dot means any release of that series, so "2.3." accepts 2.3.5.

for name, wanted in WANTED.items():
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        print(f"  missing  {{name}}")
        problems.append(f"{{name}} is not installed")
        continue
    ok = have.startswith(wanted) if wanted.endswith(".") else have == wanted
    print(f"  {{'ok     ' if ok else 'differs'}}  {{name:13}} {{have:9}} (course: {{wanted.rstrip('.')}})")
    if not ok:
        problems.append(f"{{name}} {{have}} is not the course's {{wanted.rstrip('.')}}")''')

md("## 2 · The data files\n\nThey have to sit in a `data/` folder next to this notebook, as in the ZIP.")

data_literal = "{\n" + "".join(f'    "{k}": {v},\n' for k, v in DATA.items()) + "}"
code(f'''import os
import pandas as pd

EXPECTED_ROWS = {data_literal}

if not os.path.isdir("data"):
    print("  no data/ folder here. Open the unzipped course folder in VS Code, not a single file.")
    problems.append("the data/ folder was not found next to this notebook")
else:
    for name, rows in EXPECTED_ROWS.items():
        path = os.path.join("data", name)
        if not os.path.exists(path):
            print(f"  missing  {{path}}")
            problems.append(f"{{path}} is missing")
            continue
        n = len(pd.read_csv(path))
        print(f"  {{'ok     ' if n == rows else 'differs'}}  {{name:22}} {{n:,}} rows")
        if n != rows:
            problems.append(f"{{path}} has {{n:,}} rows, not {{rows:,}}")''')

md("## 3 · Every kind of model, and a figure\n\n"
   "One fit of each model family from the course, on the index table of Session 6.")

code('''import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

if os.path.exists("data/market_features.csv"):
    table = pd.read_csv("data/market_features.csv", parse_dates=["date"]).set_index("date")
else:
    # Without the data, fit on random numbers instead: this still proves every library works.
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.normal(size=(2000, 4)), columns=["vol_5d", "vol_20d", "ret_20d", "vol_next"],
                         index=pd.date_range("2019-01-01", periods=2000, freq="D"))
    print("  (no data/market_features.csv here, so the models below fit random numbers)")
columns = list(table.columns[:-1])
train, test = table.loc[:"2022-12-31"], table.loc["2023-01-01":]
label = (train["vol_next"] > train["vol_20d"]).astype(int)

models = {
    "linear regression": LinearRegression(),
    "ridge in a pipeline": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))]),
    "a tree": DecisionTreeRegressor(max_depth=3, random_state=0),
    "a forest": RandomForestRegressor(n_estimators=50, max_features="sqrt", random_state=0),
    "boosting (scikit-learn)": GradientBoostingRegressor(n_estimators=50, max_depth=1, random_state=0),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=1, learning_rate=0.1, random_state=0),
    "LightGBM": LGBMRegressor(n_estimators=50, num_leaves=2, random_state=0, verbose=-1),
}
for name, model in models.items():
    try:
        model.fit(train[columns], train["vol_next"])
        error = np.sqrt(((test["vol_next"] - model.predict(test[columns])) ** 2).mean())
        print(f"  ok       {name:24} test RMSE {error:.3f}")
    except Exception as e:
        print(f"  FAILED   {name:24} {type(e).__name__}: {e}")
        problems.append(f"{name} did not fit: {type(e).__name__}")

for name, model in {"logistic regression": Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]),
                    "k-nearest neighbours": Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(25))])}.items():
    try:
        auc = cross_val_score(model, train[columns], label, cv=TimeSeriesSplit(5), scoring="roc_auc").mean()
        print(f"  ok       {name:24} AUC on the folds {auc:.3f}")
    except Exception as e:
        print(f"  FAILED   {name:24} {type(e).__name__}: {e}")
        problems.append(f"{name} did not fit: {type(e).__name__}")

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(table.index, table["vol_20d"])
ax.set_title("If you can see this line, figures work", loc="left")
plt.show()''')

md("## 4 · The verdict")

code('''if problems:
    print("NOT READY. Fix these, then run the notebook again:")
    for p in problems:
        print("  -", p)
    print("\\nThe setup guide on the course site shows how: a different Python, a missing package,")
    print("or a file that is not where it should be. Do it while you still have internet.")
else:
    print("ready: this laptop runs everything the course used, offline.")''')

md("""One more thing the check cannot see: **switch off AI and code completion in VS Code**,
as the practicalities guide on the Exam Info page shows, and run one mock exam in three
uninterrupted hours on this laptop.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
# Deterministic cell ids, as in the other generators, so a rebuild is not a rewrite.
for i, cell in enumerate(nb.cells):
    cell["id"] = f"c{i:04d}"
nbf.write(nb, OUT)
print(f"wrote {OUT.name} ({len(cells)} cells; pins: {', '.join(f'{k} {v}' for k, v in PINS.items())})")
