"""Shared by the tests: tools/ on the import path, and a small project built in a temporary directory."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

HEAD = 'source = "weichseltree/test"\ntitle = "Test"\nlicence = "CC0-1.0"\ncite = "nobody"\n'


def write_project(root, toml, arrays):
    """A project directory: layers.toml (HEAD + toml) and results/<name>.npy for every array."""
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "layers.toml").write_text(HEAD + toml, encoding="utf-8")
    for name, array in arrays.items():
        np.save(root / "results" / (name + ".npy"), array)
    return root


def field(rows, cols, seed=0):
    """A random field of rows x cols float64 values between -3 and 7."""
    return np.random.default_rng(seed).uniform(-3.0, 7.0, size=(rows, cols))


@pytest.fixture
def project(tmp_path):
    """One 512 x 1024 array with a patch of NaN, published as uint8 and as uint16, levels 0 and 1."""
    a = field(512, 1024)
    a[101:140, 301:420] = np.nan          # odd edges: some 2 x 2 blocks are only partly without data
    toml = "".join('''
[[layer]]
id = "%s"
title = "Field"
body = "earth"
array = "results/a.npy"
unit = "K"
type = "%s"
nodata = %d
levels = [0, 1]
''' % (kind, kind, nodata) for kind, nodata in (("uint8", 255), ("uint16", 0)))
    return write_project(tmp_path / "project", toml, {"a": a}), a
