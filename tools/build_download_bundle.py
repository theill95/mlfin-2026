# -*- coding: utf-8 -*-
"""Build downloads/mlfin-course.zip: the student bundle.

Deliberately NOT the whole repository. A GitHub source archive would hand
students the instructor README, the generators and verifiers in tools/, the
Quarto extensions, and the build machinery. This assembles the bundle from an
explicit allowlist instead, so nothing is included by accident.

    python tools/build_download_bundle.py

The archive is written with fixed timestamps, so rebuilding it after a change
that does not affect its contents produces an identical file and no git diff.
"""
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "downloads" / "mlfin-course.zip"

# Everything a student works on, and nothing else.
INCLUDE_GLOBS = [
    "session_*/session_*_exercises.ipynb",
    "session_*/session_*_case.ipynb",
    "session_*/data/*.csv",
    "data/*.csv",
]

# Anything matching these must never reach a student, whatever the globs say.
NEVER = ("tools/", "_extensions/", ".quarto/", ".git", "README.md",
         "requirements.txt", "serve.cmd", "_quarto.yml", "theme/")

FIXED_TIME = (2026, 1, 1, 0, 0, 0)     # so the zip is byte-stable

STUDENT_README = """Machine Learning in Finance
Aarhus University

This archive holds the exercise and case notebooks for every session, and the
price data they use.

    session_01/  session_02/  session_03/     the notebooks, and a copy of the
                                              data each one needs
    data/                                     all of the price files together

Getting started
---------------
1. Unzip this somewhere sensible.
2. In VS Code, use File > Open Folder and pick the unzipped folder. Open the
   folder, not a single file: that is what makes data/prices.csv resolve.
3. Open a notebook, choose your Python when asked, and run the first cell.

You need Python, VS Code and a few packages. The setup guide walks through it:

    https://theill95.github.io/mlfin-2026/setup.html

If you would rather not install anything, every notebook also opens in Google
Colab straight from the course page, and loads its data by itself:

    https://theill95.github.io/mlfin-2026/

The lectures are not in this archive. They are interactive pages whose code
cells only run when served over the web, so they are best used online, from the
course page above.

Questions: jobo@econ.au.dk

(c) 2026 Jonas Theill Bojstrup. Course materials for enrolled students.
"""


def main():
    files = []
    for pattern in INCLUDE_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))

    kept = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(bad) or rel == bad for bad in NEVER):
            print(f"  refused: {rel}", file=sys.stderr)
            continue
        kept.append((path, rel))

    if not kept:
        print("nothing matched the allowlist", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("mlfin-course/README.txt", FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, STUDENT_README)
        for path, rel in kept:
            info = zipfile.ZipInfo(f"mlfin-course/{rel}", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())

    size = OUT.stat().st_size / 1_000_000
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: "
          f"{len(kept) + 1} files, {size:.1f} MB")
    for _, rel in kept:
        print(f"    {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
