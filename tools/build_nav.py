# -*- coding: utf-8 -*-
"""Stamp the shared sidebar into every site page.

The sidebar lives in one place, assets/nav.html. This copies it into each page
between the <!-- nav:start --> and <!-- nav:end --> markers, marking the current
page as you go. Without this the nav is repeated five times and drifts the first
time somebody edits one copy.

    python tools/build_nav.py

cheatsheet.html is written by build_cheatsheet.py, which reads the same template,
so it stays in step automatically.
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "assets" / "nav.html"

PAGES = ["index.html", "setup.html", "resources.html", "downloads.html"]

START = "  <!-- nav:start -->"
END = "  <!-- nav:end -->"


def nav_for(page):
    """The shared sidebar, with `page` marked as the one being viewed."""
    nav = TEMPLATE.read_text(encoding="utf-8").rstrip("\n")
    return nav.replace(f'<a href="{page}">', f'<a href="{page}" aria-current="page">')


def main():
    changed, missing = [], []
    for page in PAGES:
        path = ROOT / page
        text = path.read_text(encoding="utf-8")
        if START not in text or END not in text:
            missing.append(page)
            continue
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        new = head + START + "\n  " + nav_for(page) + "\n" + END + tail
        if new != text:
            path.write_text(new, encoding="utf-8", newline="\n")
            changed.append(page)

    for page in changed:
        print(f"  updated {page}")
    if missing:
        print("no nav markers in: " + ", ".join(missing), file=sys.stderr)
        print("add <!-- nav:start --> and <!-- nav:end --> around the sidebar",
              file=sys.stderr)
        return 1
    print(f"nav stamped into {len(PAGES)} pages ({len(changed)} changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
