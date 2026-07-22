# -*- coding: utf-8 -*-
"""Prove the notebooks can load their data the way Google Colab will.

Colab has no local data/ folder, so every loader falls through to REPO_RAW_URL.
This runs each notebook's loader cell twice: once normally (local files) and
once with CANDIDATE_DIRS emptied, which forces the download path. The two runs
must agree, and the forced run must not raise.

    python tools/verify/colab_data_access.py
"""
import ast
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS = [
    ROOT / "session_01" / "session_01_case.ipynb",
    ROOT / "session_02" / "session_02_case.ipynb",
    ROOT / "session_03" / "session_03_case.ipynb",
    ROOT / "session_03" / "session_03_exercises.ipynb",
    ROOT / "session_04" / "session_04_case.ipynb",
    ROOT / "session_04" / "session_04_exercises.ipynb",
]


def loader_cell(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "REPO_RAW_URL" in src:
            return src
    raise AssertionError(f"no loader cell in {path.name}")


def run(src, cwd):
    """Execute a loader cell and return everything it printed."""
    out = io.StringIO()
    ns = {"__name__": "__main__"}
    import os
    here = os.getcwd()
    os.chdir(cwd)
    try:
        with redirect_stdout(out):
            exec(compile(src, "<loader>", "exec"), ns)
    finally:
        os.chdir(here)
    return out.getvalue().strip(), ns


failures = []
for nb_path in NOTEBOOKS:
    src = loader_cell(nb_path)

    url = re.search(r'REPO_RAW_URL = "([^"]+)"', src)
    if not url:
        failures.append(f"{nb_path.name}: REPO_RAW_URL is not set to a URL")
        print(f"[FAIL] {nb_path.name}: REPO_RAW_URL is not set")
        continue

    local_out, _ = run(src, nb_path.parent)

    # Empty the search path, so the only way to the data is the URL.
    forced = re.sub(r"CANDIDATE_DIRS = \[[^\]]*\]", "CANDIDATE_DIRS = []", src)
    assert "CANDIDATE_DIRS = []" in forced, nb_path.name
    try:
        remote_out, _ = run(forced, nb_path.parent)
    except Exception as exc:
        failures.append(f"{nb_path.name}: download path raised {type(exc).__name__}: {exc}")
        print(f"[FAIL] {nb_path.name}: {type(exc).__name__}: {exc}")
        continue

    if local_out != remote_out:
        failures.append(f"{nb_path.name}: local and downloaded data disagree")
        print(f"[FAIL] {nb_path.name}\n  local : {local_out}\n  remote: {remote_out}")
    else:
        first = local_out.splitlines()[0] if local_out else "(no output)"
        print(f"[ok]   {nb_path.name:32} local == downloaded   {first}")

print("=" * 74)
if failures:
    print(f"{len(failures)} problem(s):")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print(f"all {len(NOTEBOOKS)} notebooks load their data with no local files present")
