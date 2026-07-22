# -*- coding: utf-8 -*-
"""Rebuild everything that is generated, then check it, before you push.

Run this after changing any generator, any page, or any data file, and always
before publishing a new session. It is the one command that keeps the notebooks,
the cheatsheet, the sidebar and the download bundle in step with each other.

    python tools/release.py            build, then check
    python tools/release.py --check    check only, change nothing

What it does NOT do: render the Quarto decks (that needs Quarto, and is slow),
or commit anything. Both are deliberate.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BUILD = [
    ("notebooks: session 1 exercises", "tools/generators/session_01_exercises.py"),
    ("notebooks: session 1 case",      "tools/generators/session_01_case.py"),
    ("notebooks: session 2 exercises", "tools/generators/session_02_exercises.py"),
    ("notebooks: session 2 case",      "tools/generators/session_02_case.py"),
    ("notebooks: session 3 exercises", "tools/generators/session_03_exercises.py"),
    ("notebooks: session 3 case",      "tools/generators/session_03_case.py"),
    ("notebooks: session 4 exercises", "tools/generators/session_04_exercises.py"),
    ("notebooks: session 4 case",      "tools/generators/session_04_case.py"),
    ("site: sidebar",                  "tools/build_nav.py"),
    ("site: cheatsheet",               "tools/build_cheatsheet.py"),
    ("site: download bundle",          "tools/build_download_bundle.py"),
]

CHECK = [
    ("data loads the way Colab will", "tools/verify/colab_data_access.py"),
    ("session 3 exercise solutions",  "tools/verify/session_03_exercises.py"),
    ("session 3 case solutions",      "tools/verify/session_03_case.py"),
    ("session 3 deck cells",          "tools/verify/session_03_deck.py"),
    ("session 4 exercise solutions",  "tools/verify/session_04_exercises.py"),
    ("session 4 case solutions",      "tools/verify/session_04_case.py"),
    ("session 4 deck cells",          "tools/verify/session_04_deck.py"),
]


def run(label, script):
    print(f"\n--- {label}")
    result = subprocess.run([PY, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {script}", file=sys.stderr)
        return False
    return True


def main():
    check_only = "--check" in sys.argv
    steps = (CHECK if check_only else BUILD + CHECK)

    for label, script in steps:
        if not run(label, script):
            return 1

    print("\n" + "=" * 70)
    if check_only:
        print("all checks passed")
    else:
        print("everything rebuilt and checked. Review `git status`, then commit.")
        print("If you changed a .qmd, render it too:  quarto render session_NN/session_NN.qmd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
