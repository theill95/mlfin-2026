# -*- coding: utf-8 -*-
"""Delete the compiled theme files that no deck refers to any more.

Every `quarto render` compiles theme/mlfin.scss into
session_NN_files/libs/revealjs/dist/theme/quarto-<hash>.css, and the hash
changes whenever the theme changes. The old file is left behind and, because
session_NN_files/ is tracked, it gets committed. After a few theme edits each
session carries half a dozen dead stylesheets.

    python tools/prune_render_cruft.py           list what would go
    python tools/prune_render_cruft.py --delete  remove it

It only ever deletes a quarto-*.css that no session_NN.html mentions, so the
rendered decks keep working.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    delete = "--delete" in sys.argv
    referenced = set()
    for html in sorted(ROOT.glob("session_*/session_*.html")):
        referenced.update(re.findall(r"quarto-[0-9a-f]+\.css", html.read_text(encoding="utf-8")))

    if not referenced:
        print("no rendered decks found, refusing to delete anything", file=sys.stderr)
        return 1

    freed = 0
    stale = []
    for css in sorted(ROOT.glob("session_*/session_*_files/libs/revealjs/dist/theme/quarto-*.css")):
        if css.name in referenced:
            continue
        stale.append(css)
        freed += css.stat().st_size
        if delete:
            css.unlink()

    for css in stale:
        print(("deleted " if delete else "stale   ") + css.relative_to(ROOT).as_posix())
    print(f"\n{len(stale)} stale stylesheets, {freed / 1000:.0f} kB"
          + ("" if delete else "  (re-run with --delete to remove)"))
    print("in use: " + ", ".join(sorted(referenced)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
