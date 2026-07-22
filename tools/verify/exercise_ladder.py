# -*- coding: utf-8 -*-
"""Check the exercise ladder across all four sessions.

The badges are the student's map of the notebook, and they rot quietly: a level
typed as 6, an id used twice, a `revisits` tag pointing at a session that has
not happened yet. None of that stops a notebook running, so nothing else
catches it.

    python tools/verify/exercise_ladder.py

Fails if any of these is true:
  - two exercises share an id, or a section's ids are not consecutive from 1
  - a star count is outside 1 to 5
  - a `revisits` tag names the current session or a later one
  - a session uses fewer than four of the five levels

It also prints the per-session histogram, which is the quickest way to see
whether a notebook has drifted into being all one difficulty.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ["01", "02", "03", "04"]

HEAD = re.compile(r"^### ([A-Z])(\d+) · (.+?)\s+([★☆]{5})(?:\s+· revisits (S\d))?\s*$")

failures = []
print(f"{'session':>8}  {'n':>3}  {'1*':>4}{'2*':>4}{'3*':>4}{'4*':>4}{'5*':>4}   revisits")
print("-" * 70)

for session in SESSIONS:
    path = ROOT / f"session_{session}" / f"session_{session}_exercises.ipynb"
    if not path.exists():
        failures.append(f"session {session}: notebook missing")
        continue

    nb = json.loads(path.read_text(encoding="utf-8"))
    hist = defaultdict(int)
    by_section = defaultdict(list)
    seen = set()
    tags = defaultdict(int)

    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        first = "".join(cell["source"]).split("\n")[0]
        if not first.startswith("### ") or not re.match(r"^### [A-Z]\d+ ", first):
            continue

        m = HEAD.match(first)
        if not m:
            failures.append(f"session {session}: unparsable heading -> {first[:60]}")
            continue

        letter, number, _title, stars, revisits = m.groups()
        sid = f"{letter}{number}"
        level = stars.count("★")

        if sid in seen:
            failures.append(f"session {session}: duplicate id {sid}")
        seen.add(sid)

        if not 1 <= level <= 5:
            failures.append(f"session {session} {sid}: level {level} outside 1-5")
        hist[level] += 1
        by_section[letter].append(int(number))

        if revisits:
            # a tag may only point backwards: S2 may revisit S1, never S2 or S3
            if int(revisits[1:]) >= int(session):
                failures.append(
                    f"session {session} {sid}: revisits {revisits}, which is not "
                    f"an earlier session")
            tags[revisits] += 1

    for letter, numbers in by_section.items():
        expected = list(range(1, len(numbers) + 1))
        if sorted(numbers) != expected:
            failures.append(
                f"session {session}: section {letter} ids are {sorted(numbers)}, "
                f"expected {expected}")

    used = sum(1 for level in range(1, 6) if hist[level])
    if used < 4:
        failures.append(
            f"session {session}: only {used} of the five levels are used")

    tag_text = ", ".join(f"{k}:{v}" for k, v in sorted(tags.items())) or "none"
    print(f"{session:>8}  {len(seen):>3}  " +
          "".join(f"{hist[level]:>4}" for level in range(1, 6)) +
          f"   {tag_text}")

print("-" * 70)
if failures:
    print(f"\n{len(failures)} problem(s):")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("ladder is sound: ids unique and consecutive, levels in range, "
      "revisits tags all point backwards")
