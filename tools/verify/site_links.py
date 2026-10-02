"""Every link inside the site leads somewhere: the file exists, and so does the
anchor after its #.

Checks the site's pages, the links the decks make, and every entry of the
search index. A link to a slide, deck.html#/slide-id, needs a slide with that
id. Colab links and links to the published site must name a file in this
repository. Other external links are not fetched.

    python tools/verify/site_links.py
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SITE = "https://theill95.github.io/mlfin-2026/"
REPO_FILE = re.compile(r"^https://colab\.research\.google\.com/github/theill95/mlfin-2026/blob/main/(.+)$")
SCRIPT = re.compile(r"<script\b.*?</script\s*>", re.S | re.I)
ATTR = re.compile(r'\s(?:href|src)="([^"]*)"')

_ids = {}


def ids_of(path):
    """The ids a page has, and for a deck the ids of its slides as #/id."""
    if path not in _ids:
        text = path.read_text(encoding="utf-8", errors="replace")
        found = set(re.findall(r'\sid="([^"]+)"', text)) | set(re.findall(r'\sname="([^"]+)"', text))
        found |= {"/" + s for s in re.findall(r'<section id="([^"]+)"', text)}
        _ids[path] = found
    return _ids[path]


def check(url, base_dir, where, problems):
    """One link, seen on `where`, resolved against `base_dir`."""
    url = url.strip()
    if not url or url.startswith(("mailto:", "javascript:", "data:", "tel:")):
        return
    m = REPO_FILE.match(url)
    if m:
        if not (ROOT / unquote(m.group(1))).is_file():
            problems.append(f"{where}: Colab link to a file that is not here: {m.group(1)}")
        return
    if url.startswith(SITE):
        url, base_dir = url[len(SITE):] or "index.html", ROOT
    elif url.startswith("/mlfin-2026/"):
        url, base_dir = url[len("/mlfin-2026/"):] or "index.html", ROOT
    elif re.match(r"^[a-z]+:", url) or url.startswith("//"):
        return
    parts = urlsplit(url)
    if parts.path:
        target = (base_dir / unquote(parts.path)).resolve()
        if target.is_dir():
            target = target / "index.html"
    else:
        target = (ROOT / where).resolve() if not where.startswith("search index") else None
    if target is None:
        problems.append(f"{where}: link with no page: {url}")
        return
    if not target.is_file():
        problems.append(f"{where}: missing file {url}")
        return
    frag = unquote(parts.fragment)
    if frag and target.suffix == ".html" and frag not in ids_of(target):
        problems.append(f"{where}: no #{frag} in {target.relative_to(ROOT).as_posix()}")


def main():
    problems, n_links = [], 0
    pages = sorted(p for p in ROOT.glob("*.html"))
    pages += [ROOT / "session_07" / "map.html"]
    pages += sorted(ROOT.glob("session_[0-9][0-9]/session_[0-9][0-9].html"))
    for page in pages:
        text = SCRIPT.sub(" ", page.read_text(encoding="utf-8"))
        base = re.search(r'<base href="([^"]+)"', text)
        # The 404 page is served at any address, so its links start from the site root.
        base_dir = ROOT if base else page.parent
        where = page.relative_to(ROOT).as_posix()
        for url in ATTR.findall(text):
            n_links += 1
            check(url, base_dir, where, problems)

    index = json.loads((ROOT / "assets" / "search-index.json").read_text(encoding="utf-8"))
    for item in index["items"]:
        n_links += 1
        check(item["u"], ROOT, f"search index ({item['t'][:40]})", problems)

    if problems:
        for p in problems:
            print("  problem:", p)
        print(f"{len(problems)} broken of {n_links} links")
        sys.exit(1)
    print(f"all {n_links} links resolve ({len(pages)} pages and {len(index['items'])} search entries)")


if __name__ == "__main__":
    main()
