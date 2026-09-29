"""Check that published lecture cells match their source and start in order."""

import base64
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SETUP = (ROOT / "_extensions/r-wasm/live/templates/pyodide-setup.ojs").read_text(
    encoding="utf-8"
)
CELL = re.compile(r'^```\{pyodide\}\s*\n(.*?)^```', re.M | re.S)
SCRIPT = re.compile(r'<script type="pyodide-(\d+)-contents">\s*([^<]+?)\s*</script>')
MODULE = re.compile(r'<script type="ojs-module-contents">\s*([^<]+?)\s*</script>')


def decode(encoded):
    return json.loads(base64.b64decode(encoded))


total = 0
for source in sorted(ROOT.glob("session_*/session_*.qmd")):
    html_path = source.with_suffix(".html")
    if not html_path.exists():
        raise AssertionError(f"Missing published lecture: {html_path}")
    qmd = source.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    frontmatter = qmd.split("\n---\n", 1)[0]
    assert re.search(r"(?m)^\s+autorun: true\s*$", frontmatter), source

    chunks = list(CELL.finditer(qmd))
    published = [(int(n), decode(payload)) for n, payload in SCRIPT.findall(html)]
    assert len(chunks) == len(published), (source, len(chunks), len(published))
    assert [n for n, _ in published] == list(range(1, len(chunks) + 1)), source

    for index, (match, (_, block)) in enumerate(zip(chunks, published), 1):
        chunk = match.group(1)
        code = "\n".join(line for line in chunk.splitlines() if not line.startswith("#|"))
        assert block["code"].strip() == code.strip(), (source, index, "stale code")
        options = dict(re.findall(r"(?m)^#\|\s*(autorun):\s*(true|false)\s*$", chunk))
        expected = options.get("autorun", "true") == "true"
        assert block["attr"]["autorun"] is expected, (source, index)
        if not expected:
            before = qmd[: match.start()]
            headings = re.findall(r"(?m)^# (.+)$", before)
            heading = headings[-1] if headings else ""
            assert any(tag in heading for tag in ("[predict]", "Your turn", "[optional]")), (
                source, index, heading
            )

    modules = [decode(payload) for payload in MODULE.findall(html)]
    cells = [cell for module in modules for cell in module["contents"]]
    prelude = [cell for cell in cells if cell["cellName"] == "pyodide-prelude"]
    assert len(prelude) == 1 and prelude[0]["source"].replace("\r\n", "\n") == SETUP, source
    for index in range(1, len(chunks) + 1):
        definitions = [cell for cell in cells if cell["cellName"] == f"pyodide-{index}"]
        assert len(definitions) == 1, (source, index)
        assert re.search(rf"pyodideOjs\.process\([^\n;]+, {index}\);", definitions[0]["source"]), (
            source, index
        )
    total += len(chunks)
    print(f"{source.parent.name}: {len(chunks)} cells checked")

print(f"{total} published lecture cells have matching autorun settings and ordered first runs")
