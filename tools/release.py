# -*- coding: utf-8 -*-
"""Rebuild everything that is generated, then check it, before you push.

Run this after changing any generator, any page, or any data file, and always
before publishing a new session. It is the one command that keeps the notebooks,
the cheatsheet, the sidebar and the download bundle in step with each other.

    python tools/release.py            build, then check
    python tools/release.py --check    check only, change nothing

It runs under the repository's .venv when that folder exists (see
tools/requirements-dev.txt), so the notebooks are generated with exactly the
package versions students install from requirements.txt. Generated text quotes
measured numbers, and a few of them move between library versions; the first
check stops the run if the environment does not match the pins.

What it does NOT do: render the Quarto decks (that needs Quarto, and is slow),
or commit anything. Both are deliberate.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Re-run under the course environment if it exists and this is not it.
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve() and not os.environ.get("MLFIN_NO_VENV"):
    print(f"using {VENV_PY.relative_to(ROOT)}")
    sys.exit(subprocess.run([str(VENV_PY), __file__, *sys.argv[1:]]).returncode)

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
    ("notebooks: session 5 exercises", "tools/generators/session_05_exercises.py"),
    ("notebooks: session 5 case",      "tools/generators/session_05_case.py"),
    ("data: session 6 table",          "tools/build_session06_table.py"),
    ("notebooks: session 6 exercises", "tools/generators/session_06_exercises.py"),
    ("notebooks: session 6 case",      "tools/generators/session_06_case.py"),
    ("notebooks: session 8 exercises", "tools/generators/session_08_exercises.py"),
    ("notebooks: session 8 case",      "tools/generators/session_08_case.py"),
    ("notebooks: session 9 exercises", "tools/generators/session_09_exercises.py"),
    ("notebooks: session 9 case",      "tools/generators/session_09_case.py"),
    ("notebooks: session 11 exercises", "tools/generators/session_11_exercises.py"),
    ("notebooks: session 11 case",     "tools/generators/session_11_case.py"),
    ("notebooks: session 12 exercises", "tools/generators/session_12_exercises.py"),
    ("notebooks: session 12 case",     "tools/generators/session_12_case.py"),
    ("notebook: setup check",          "tools/generators/setup_check.py"),
    ("site: revision page",            "tools/build_revision.py"),
    ("site: sidebar",                  "tools/build_nav.py"),
    ("site: cheatsheet",               "tools/build_cheatsheet.py"),
    ("site: PDFs of the lectures and pages", "tools/build_handouts.py"),
    ("site: download bundle",          "tools/build_download_bundle.py"),
]

CHECK = [
    ("environment matches requirements.txt", "tools/verify/environment.py"),
    ("published lecture autorun", "tools/verify/lecture_autorun.py"),
    ("earlier lecture cell state", "tools/verify/lecture_cell_state.py"),
    ("data loads the way Colab will", "tools/verify/colab_data_access.py"),
    ("the exercise ladder",           "tools/verify/exercise_ladder.py"),
    ("blank Run-all is safe",        "tools/verify/blank_safety.py"),
    ("markdown Colab can render",     "tools/verify/colab_markup.py"),
    ("session 1 exercise solutions",  "tools/verify/session_01_exercises.py"),
    ("session 1 case solutions",      "tools/verify/session_01_case.py"),
    ("session 2 exercise solutions",  "tools/verify/session_02_exercises.py"),
    ("session 2 case solutions",      "tools/verify/session_02_case.py"),
    ("session 3 exercise solutions",  "tools/verify/session_03_exercises.py"),
    ("session 3 case solutions",      "tools/verify/session_03_case.py"),
    ("session 3 deck cells",          "tools/verify/session_03_deck.py"),
    ("session 4 exercise solutions",  "tools/verify/session_04_exercises.py"),
    ("session 4 case solutions",      "tools/verify/session_04_case.py"),
    ("session 4 deck cells",          "tools/verify/session_04_deck.py"),
    ("session 5 exercise solutions",  "tools/verify/session_05_exercises.py"),
    ("session 5 case solutions",      "tools/verify/session_05_case.py"),
    ("session 5 deck cells",          "tools/verify/session_05_deck.py"),
    ("session 6 exercise solutions",  "tools/verify/session_06_exercises.py"),
    ("session 6 case solutions",      "tools/verify/session_06_case.py"),
    ("session 6 deck cells",          "tools/verify/session_06_deck.py"),
    ("session 7 deck cells",          "tools/verify/session_07_deck.py"),
    ("session 8 exercise solutions",  "tools/verify/session_08_exercises.py"),
    ("session 8 case solutions",      "tools/verify/session_08_case.py"),
    ("session 8 deck cells",          "tools/verify/session_08_deck.py"),
    ("session 9 exercise solutions",  "tools/verify/session_09_exercises.py"),
    ("session 9 case solutions",      "tools/verify/session_09_case.py"),
    ("session 9 deck cells",          "tools/verify/session_09_deck.py"),
    ("session 10 deck cells",         "tools/verify/session_10_deck.py"),
    ("session 11 exercise solutions", "tools/verify/session_11_exercises.py"),
    ("session 11 case solutions",     "tools/verify/session_11_case.py"),
    ("session 11 deck cells",         "tools/verify/session_11_deck.py"),
    ("session 12 exercise solutions", "tools/verify/session_12_exercises.py"),
    ("session 12 case solutions",     "tools/verify/session_12_case.py"),
    ("session 12 deck cells",         "tools/verify/session_12_deck.py"),
    ("session 13 deck cells",         "tools/verify/session_13_deck.py"),
    ("the setup check says ready",    "tools/verify/setup_check.py"),
    ("the PDFs match their sources", "tools/verify/handouts.py"),
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
    # The environment check comes first either way: building with other library
    # versions would quietly change the numbers the notebooks quote.
    steps = (CHECK if check_only else CHECK[:1] + BUILD + CHECK[1:])

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
