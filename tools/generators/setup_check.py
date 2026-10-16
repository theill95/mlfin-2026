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

md("""# Exam laptop readiness check

**Updated 15 October 2026. Everyone: run this updated check before the exam.**
Mac users: pay particular attention to XGBoost and LightGBM. Some installations need
an extra library; **if all checks pass, no additional installation is needed**.

1. Download and unzip the latest course ZIP, or replace `setup_check.ipynb` in your
   existing unzipped course folder with this updated notebook.
2. In **local VS Code**, open that folder (with `data/` beside this notebook).
3. Select the **same local Python kernel you will use for the exam**.
4. Turn off Wi-Fi and any other internet connection, **restart the kernel**, then
   **Run All**. Read the final verdict and check that the plot is visible.

A run in Colab, a remote kernel, or another Python environment does not test your
exam setup. This notebook never downloads or installs anything. A pass covers these
checks, not every possible exam problem; also rehearse a mock exam offline.

If anything fails, reconnect to fix it using the instructions below, then restart
the kernel and repeat **all** checks offline. Do not run just the final cell.""")

code(r'''import sys
import platform

problems = []
completed = set()
required = {"Python", "packages", "data", "linear regression", "ridge in a pipeline",
            "a tree", "a forest", "boosting (scikit-learn)", "XGBoost", "LightGBM",
            "logistic regression", "k-nearest neighbours", "figure"}

print("Readiness check: 2026-10-15")
print("Python:", sys.version.split()[0])
print("Python location:", sys.executable)
print("System:", platform.system(), platform.release(), "Architecture:", platform.machine())
print("Run locally in VS Code, with internet off. This check cannot verify those steps for you.")

def repair_hint(error):
    message = str(error).lower()
    if platform.system() == "Darwin" and "libomp" in message and any(
        word in message for word in ("not loaded", "not found", "no such file", "image not found")
    ) and "incompatible architecture" not in message:
        return ("macOS: the error indicates a missing OpenMP library. While online, install "
                "Homebrew from https://brew.sh if needed and follow its Next steps. "
                "In macOS Terminal run: brew install libomp. Restart the notebook kernel, "
                "then Run All offline. See the repair instructions below.")
    if "incompatible architecture" in message or "wrong architecture" in message:
        return ("Architecture mismatch: Python and its native libraries must match. "
                "On Apple Silicon use native arm64 Python and Homebrew together; do not mix "
                "an Intel/Rosetta Python with ARM libraries. Ask for help with the full error.")
    if isinstance(error, ModuleNotFoundError):
        return ("Missing package: while online, run the course %pip install command below "
                "in this notebook's selected kernel. Restart, then Run All offline.")
    return ("Keep the full error and the environment details above. Check Exam info / Setup & tools; "
            "ask for help before exam day if it persists. Do not change packages that already pass.")

def check(name, action):
    # Only a completed, successful action can pass. Exceptions never disappear into a skip.
    completed.discard(name)
    try:
        action()
    except Exception as error:
        problems.append(f"{name}: {type(error).__name__}: {error}")
        print(f"FAILED {name}: {type(error).__name__}: {error}")
        print(repair_hint(error))
    else:
        completed.add(name)
        print("PASS", name)

def check_python():
    if not ((3, 12) <= sys.version_info[:2] <= (3, 14)):
        raise RuntimeError("Choose Python 3.12, 3.13 or 3.14 (3.14 recommended), then install the course packages.")

check("Python", check_python)''')

md("## 1 · Installed package versions\n\nThese checks do not prove a library can load or run; the model checks below do that.")
code("WANTED = " + repr(PINS) + r'''
from importlib import metadata

def check_packages():
    failures = []
    for name, wanted in WANTED.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            failures.append(f"{name} is not installed")
            continue
        ok = have.startswith(wanted) if wanted.endswith(".") else have == wanted
        print(f"  {name}: {have} (course: {wanted.rstrip('.')})")
        if not ok:
            failures.append(f"{name} {have} is not the course's {wanted.rstrip('.')}")
    if failures:
        raise RuntimeError("; ".join(failures) + ". While online, use the course %pip install command below in this kernel, then restart.")

check("packages", check_packages)''')

md("## 2 · Local data\n\nMissing or unreadable files are reported, never downloaded.")
code("EXPECTED_ROWS = " + repr(DATA) + r'''
from pathlib import Path
import csv

def check_data():
    failures = []
    for name, rows in EXPECTED_ROWS.items():
        try:
            with (Path("data") / name).open(encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                n = sum(1 for _ in reader)
            if n != rows:
                raise ValueError(f"{n} rows, expected {rows}")
            print(f"  data/{name}: {n:,} rows")
        except Exception as error:
            failures.append(f"data/{name}: {error}")
    if failures:
        raise RuntimeError("; ".join(failures) + ". Unzip a fresh course ZIP and open its folder in VS Code, with data/ beside this notebook.")

check("data", check_data)''')

md("## 3 · Import, fit and predict\n\nEach model is checked independently, including XGBoost and LightGBM. A failure in one does not hide the other's result.")
code(r'''def model_data():
    import numpy as np
    import pandas as pd
    path = Path("data/market_features.csv")
    if path.exists():
        table = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    else:
        # Still test libraries if data is missing; the separate data check remains failed.
        rng = np.random.default_rng(0)
        table = pd.DataFrame(rng.normal(size=(2000, 4)),
                             columns=["vol_5d", "vol_20d", "ret_20d", "vol_next"],
                             index=pd.date_range("2019-01-01", periods=2000, freq="D"))
    return table, table.loc[:"2022-12-31"], table.loc["2023-01-01":]

def check_model(name):
    import numpy as np
    # Import inside the protected action: an installed package can still fail to load.
    if name == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=50, max_depth=1, learning_rate=0.1, random_state=0, n_jobs=2)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=50, num_leaves=2, random_state=0, verbose=-1, n_jobs=2)
    else:
        from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        model = {
            "linear regression": lambda: LinearRegression(),
            "ridge in a pipeline": lambda: Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))]),
            "a tree": lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
            "a forest": lambda: RandomForestRegressor(n_estimators=50, max_features="sqrt", random_state=0),
            "boosting (scikit-learn)": lambda: GradientBoostingRegressor(n_estimators=50, max_depth=1, random_state=0),
            "logistic regression": lambda: Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]),
            "k-nearest neighbours": lambda: Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(25))]),
        }[name]()
    table, train, test = model_data()
    columns = list(table.columns[:-1])
    if name in {"logistic regression", "k-nearest neighbours"}:
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        label = (train["vol_next"] > train["vol_20d"]).astype(int)
        scores = cross_val_score(model, train[columns], label, cv=TimeSeriesSplit(5), scoring="roc_auc", error_score="raise")
        if not np.isfinite(scores).all():
            raise ValueError("The cross-validation scores are not finite")
        print(f"  AUC on the folds: {scores.mean():.3f}")
    else:
        model.fit(train[columns], train["vol_next"])
        prediction = np.asarray(model.predict(test[columns]))
        if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
            raise ValueError("Predictions have the wrong shape or contain non-finite values")
        print(f"  test RMSE: {np.sqrt(((test['vol_next'] - prediction) ** 2).mean()):.3f}")

for name in ["linear regression", "ridge in a pipeline", "a tree", "a forest",
             "boosting (scikit-learn)", "XGBoost", "LightGBM",
             "logistic regression", "k-nearest neighbours"]:
    check(name, lambda name=name: check_model(name))''')

code(r'''def check_figure():
    import matplotlib.pyplot as plt
    table, _, _ = model_data()
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(table.index, table["vol_20d"])
    ax.set_title("Check that you can see this line", loc="left")
    plt.show()

check("figure", check_figure)''')

md("## 4 · The verdict\n\nIf execution stops or the kernel crashes, the check has **not passed**. Restart and Run All; ask for help if it happens again.")
code(r'''if "required" not in globals() or "completed" not in globals() or "problems" not in globals():
    print("NOT READY: checks have not run. Restart the kernel and Run All.")
else:
    incomplete = required - completed
    if problems or incomplete:
        print("NOT READY. Fix the reported issues, then restart the kernel and Run All offline.")
        for problem in problems:
            print("  -", problem)
        if incomplete:
            print("  Checks that did not pass:", ", ".join(sorted(incomplete)))
        print("Repair instructions are below; keep the full errors and environment details if you need help.")
    else:
        print("READY: all automated checks passed in this Python environment.")
        print("Confirm you ran locally in VS Code with internet off and can see the plot.")
        print("No additional installation is needed. Practise one mock exam offline as well.")''')

md("""## If a check fails

**Only fix a problem the check reports. If everything passes, do not change your setup.**

- **Missing package or wrong version:** reconnect and run the following in a notebook
  cell using your exam kernel (not Terminal), then restart the kernel:

  ```python
  %pip install -r https://theill95.github.io/mlfin-2026/requirements.txt
  ```

- **Mac: an error says `libomp.dylib` is not loaded/found:** this is the OpenMP runtime,
  separate from the Python packages. While online, install [Homebrew](https://brew.sh)
  if needed and follow the installer's **Next steps** (including its PATH setup).
  Then run this in **macOS Terminal**, not a Python cell:

  ```sh
  brew install libomp
  ```

  Restart the notebook kernel and repeat Run All offline. If `brew` is not found,
  finish Homebrew's Next steps and reopen Terminal. Installing `libomp` with pip is
  not the fix. If the library is already installed and the error persists, ask for
  help with the full error instead of repeatedly reinstalling packages.

- **Mac: incompatible architecture:** Python and the native libraries must match.
  On Apple Silicon, prefer native arm64 Python and Homebrew; avoid mixing an
  Intel/Rosetta Python with ARM libraries. Ask for help with the environment details
  printed above. Existing Conda users should repair their Conda environment with
  help rather than mix package managers or replace a working environment.
- **Missing/unreadable data:** unzip a fresh course ZIP and use File → Open Folder
  in VS Code. Keep this notebook beside `data/`.
- **Any other error or kernel crash:** save the full error and the environment
  details above and ask for help before the exam. A crash is not a pass.

After any repair, restart the kernel and Run All again **offline**. A successful
Colab run does not establish that your local Mac or Windows laptop is ready.
See [Exam info](https://theill95.github.io/mlfin-2026/exam_info.html#readiness)
and [Setup & tools](https://theill95.github.io/mlfin-2026/setup.html#macos).

Also switch off AI completion in VS Code as the practicalities guide explains (the
suggestion list that offers `head` after `df.` is allowed and stays on), and rehearse
a mock exam on this laptop. This check does not verify editor settings or
guarantee that every possible exam task will work.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
# Deterministic cell ids, as in the other generators, so a rebuild is not a rewrite.
for i, cell in enumerate(nb.cells):
    cell["id"] = f"c{i:04d}"
nbf.write(nb, OUT)
print(f"wrote {OUT.name} ({len(cells)} cells; pins: {', '.join(f'{k} {v}' for k, v in PINS.items())})")
