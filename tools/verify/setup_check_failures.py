"""Exercise readiness failure paths, including native macOS import failures.

The real notebook's code is executed with injected faults; no student files or
installed packages are changed. The normal notebook verifier covers the real kernel.
"""
import builtins
import contextlib
import io
import os
import socket
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parents[2]
CELLS = [c.source for c in nbformat.read(ROOT / 'setup_check.ipynb', as_version=4).cells if c.cell_type == 'code']
REAL_IMPORT = builtins.__import__

def run(fault=None, skip=None):
    ns = {}
    output = io.StringIO()
    def importer(name, *args, **kwargs):
        if name == 'xgboost' and fault == 'xgb_libomp':
            raise ValueError('Library not loaded: @rpath/libomp.dylib (image not found)')
        if name == 'lightgbm' and fault == 'lgb_libomp':
            raise OSError('Library not loaded: libomp.dylib (no such file)')
        if name == 'xgboost' and fault == 'architecture':
            raise OSError('libomp.dylib: incompatible architecture')
        if name == 'xgboost' and fault == 'missing_package':
            raise ModuleNotFoundError("No module named 'xgboost'")
        return REAL_IMPORT(name, *args, **kwargs)
    real_version = metadata.version
    def version(name):
        return '0.0.0' if name == 'lightgbm' and fault == 'version' else real_version(name)
    real_open = Path.open
    def file_open(path, *args, **kwargs):
        if path.as_posix() == 'data/prices.csv' and fault == 'data':
            raise FileNotFoundError('missing data/prices.csv')
        return real_open(path, *args, **kwargs)
    with contextlib.redirect_stdout(output), patch('builtins.__import__', side_effect=importer), \
         patch.object(metadata, 'version', side_effect=version), \
         patch.object(Path, 'open', file_open), patch.object(plt, 'show'), \
         patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden in readiness check')):
        for i, source in enumerate(CELLS):
            if i != skip:
                exec(compile(source, f'notebook-cell-{i}', 'exec'), ns)
                if i == 0:
                    # Mock only the notebook's diagnostic OS, not libraries' binary loaders.
                    ns['platform'] = SimpleNamespace(system=lambda: 'Darwin')
        plt.close('all')
    return output.getvalue(), ns

def main():
    os.chdir(ROOT)
    text, ns = run()
    assert 'READY: all automated checks passed' in text and not ns['problems'], text
    print('PASS: all real checks, with network blocked')
    for fault in ('xgb_libomp', 'lgb_libomp', 'architecture', 'missing_package', 'version', 'data'):
        text, ns = run(fault)
        assert 'NOT READY' in text and 'READY: all automated' not in text, text
        if fault == 'xgb_libomp':
            assert 'PASS LightGBM' in text and 'brew install libomp' in text, text
        if fault == 'lgb_libomp':
            assert 'PASS XGBoost' in text and 'brew install libomp' in text, text
        if fault == 'architecture':
            assert 'Architecture mismatch' in text and 'brew install libomp' not in text, text
        if fault == 'missing_package':
            assert 'Missing package' in text and 'brew install libomp' not in text, text
        print('PASS:', fault)
    text, _ = run(skip=3)  # skip all model execution
    assert 'NOT READY' in text and 'READY: all automated' not in text, text
    print('PASS: skipped checks cannot pass')
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(CELLS[-1], {})
    assert 'NOT READY' in out.getvalue()
    print('PASS: verdict alone cannot pass')
    # A runtime error and NaN predictions are failures, not just import checks.
    for prediction in ('raises', 'nan'):
        text, ns = run()
        ns['completed'].discard('XGBoost')
        import xgboost
        import numpy as np
        with patch.object(xgboost.XGBRegressor, 'predict', side_effect=RuntimeError('prediction failed') if prediction == 'raises' else lambda x: np.full(len(x), np.nan)), contextlib.redirect_stdout(io.StringIO()):
            ns['check']('XGBoost', lambda: ns['check_model']('XGBoost'))
        assert 'XGBoost' not in ns['completed'] and ns['problems']
        print('PASS: prediction', prediction)

if __name__ == '__main__':
    main()
