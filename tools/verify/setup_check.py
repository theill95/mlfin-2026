# -*- coding: utf-8 -*-
"""Run setup_check.ipynb in the course environment and require its verdict.

The notebook is what students run on their exam laptop. Here it runs from the top
of the repository, where data/ sits next to it exactly as in the course ZIP, and
it must end with "ready". Run under .venv (release.py does), or the version
checks will rightly say not ready.

    python tools/verify/setup_check.py
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
nb = nbformat.read(ROOT / "setup_check.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()

text = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
               for o in c.get("outputs", []) if o.get("name") == "stdout")
errors = [o["ename"] for c in nb.cells if c.cell_type == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
figures = sum(1 for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "display_data")
print(text.strip().splitlines()[-1])
if errors or figures != 1 or "ready: this laptop" not in text:
    print(f"setup check failed: errors {errors}, figures {figures}")
    print(text)
    sys.exit(1)
print("setup_check.ipynb runs clean and says ready")
