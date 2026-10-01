# -*- coding: utf-8 -*-
"""Pieces of notebook source that several generators share.

Import from a generator with `from _shared import XGB_GUARD`: Python puts a
script's own folder on sys.path, so this works when a generator is run as
`python tools/generators/session_12_exercises.py`.
"""

# The first lines of a setup cell in a notebook that uses XGBoost. The numbers in
# Session 12 were measured with XGBoost 2.1.4, the version Pyodide runs in the
# lecture. XGBoost 3 bins the columns differently (base_score and max_bin do not
# change that; tree_method="exact" agrees across versions), so in Colab, which
# ships XGBoost 3, about a third of the session's XGBoost results came out
# different. The guard installs 2.1.4 there before xgboost is first imported, and
# anywhere else only says which version the numbers came from.
XGB_GUARD = '''# The numbers in this notebook come from XGBoost 2.1.4, the version the lecture runs. Colab
# ships a newer XGBoost that grows slightly different trees, so there the first run installs
# 2.1.4 (about 20 seconds, once per Colab session).
import sys
import importlib.metadata
try:
    xgboost_version = importlib.metadata.version("xgboost")
except importlib.metadata.PackageNotFoundError:
    xgboost_version = None
if xgboost_version != "2.1.4":
    if "google.colab" in sys.modules:
        import subprocess
        print("Installing XGBoost 2.1.4 for this Colab session...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost==2.1.4"], check=True)
    else:
        print(f"Note: you have XGBoost {xgboost_version}; the numbers here come from 2.1.4, "
              "which the setup guide's install line gives you.")
'''
