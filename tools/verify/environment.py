# -*- coding: utf-8 -*-
"""Check that this Python has exactly the package versions requirements.txt pins.

The generators measure numbers while they build the notebooks (a forest's error,
the trees early stopping keeps) and write them into the text. A few of those move
between library versions: scikit-learn 1.6 changed some forest results in the
fourth decimal, and XGBoost 3 grows different trees from 2.1. Building with the
versions students install keeps the text and what students see in step, so
release.py runs this before anything else.

    python tools/verify/environment.py
"""
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_PYTHON = ((3, 12), (3, 14))   # the versions requirements.txt has installers for

problems = []
print(f"Python {sys.version.split()[0]} at {sys.executable}")
if not (SUPPORTED_PYTHON[0] <= sys.version_info[:2] <= SUPPORTED_PYTHON[1]):
    problems.append(f"Python {sys.version_info[0]}.{sys.version_info[1]} is outside 3.12 to 3.14")

for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        continue
    req = Requirement(line)
    try:
        have = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        problems.append(f"{req.name} is not installed (wanted {req.specifier})")
        continue
    ok = req.specifier.contains(have, prereleases=True)
    print(f"  {'ok ' if ok else 'BAD'} {req.name:14s} {have:10s} wanted {req.specifier}")
    if not ok:
        problems.append(f"{req.name} {have} does not match {req.specifier}")

if problems:
    print("\nThis environment does not match requirements.txt:")
    for p in problems:
        print("  -", p)
    print("\nCreate the course environment once, and release.py will use it by itself:\n"
          "    python -m venv .venv\n"
          r"    .venv\Scripts\python -m pip install -r tools/requirements-dev.txt")
    sys.exit(1)
print("environment matches requirements.txt")
