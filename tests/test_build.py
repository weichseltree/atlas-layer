"""build.py: what is in a tile is what was in the array, at every level, and the same input gives the same bytes."""
import json
import warnings

import numpy as np
import pytest
from PIL import Image

from build import BuildError, build
from conftest import field, write_project

QUIET = lambda line: None


def read_tile(dist, layer, level, x, y, step=None):
    """(stored integers, quantities with NaN for nodata) of one tile."""
    names = {"level": level, "x": x, "y": y}
    if step is not None:
        names[layer["axes"][0]["id"]] = step
    path = layer["files"]
    for name, value in names.items():
        path = path.replace("{%s}" % name, str(value))
    image = Image.open(dist / path)
    assert image.size == (256, 256)
    assert image.mode == ("L" if layer["values"]["type"] == "uint8" else "I;16")
    stored = np.array(image)
    quantity = stored * layer["values"]["scale"] + layer["values"]["offset"]
    if layer["values"]["nodata"] is not None:
        quantity[stored == layer["values"]["nodata"]] = np.nan
    return stored, quantity


def close(quantity, expected, layer):
    """Equal within half a quantisation step, with NaN in the same places."""
    step = abs(layer["values"]["scale"])
    assert np.array_equal(np.isnan(quantity), np.isnan(expected))
    return np.nanmax(np.abs(quantity - expected)) <= step / 2 * (1 + 1e-9)


def test_tile_equals_the_block_it_was_cut_from(project):
    root, a = project
    manifest = build(root, say=QUIET)
    for layer in manifest["layers"]:
        assert layer["grid"]["levels"] == [0, 1]
        for x, y in ((0, 0), (1, 0), (3, 1)):
            _, quantity = read_tile(root / "dist", layer, 1, x, y)
            assert close(quantity, a[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256], layer)
    # the two types differ in precision: 255 steps against 65535
    assert manifest["layers"][0]["values"]["scale"] > 200 * manifest["layers"][1]["values"]["scale"]


def test_coarser_level_is_the_block_mean(project):
    root, a = project
    manifest = build(root, say=QUIET)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # "mean of empty slice": the blocks of only NaN
        mean = np.nanmean(a.reshape(256, 2, 512, 2), axis=(1, 3))
    for layer in manifest["layers"]:
        for x in (0, 1):
            _, quantity = read_tile(root / "dist", layer, 0, x, 0)
            assert close(quantity, mean[:, x * 256:(x + 1) * 256], layer)


def test_nodata_survives(project):
    root, a = project
    manifest = build(root, say=QUIET)
    for layer in manifest["layers"]:
        nodata = layer["values"]["nodata"]
        stored, _ = read_tile(root / "dist", layer, 1, 1, 0)      # columns 256..511, rows 0..255
        assert np.array_equal(stored == nodata, np.isnan(a[0:256, 256:512]))
        assert (stored == nodata).sum() == 39 * 119
        coarse, _ = read_tile(root / "dist", layer, 0, 0, 0)
        # a coarse cell has no data only where all four cells under it had none: rows 102..139, columns 302..419
        expected = np.zeros((256, 256), dtype=bool)
        expected[51:70, 151:210] = True
        assert np.array_equal(coarse == nodata, expected)


def test_small_array_is_shown_cell_by_cell_and_levels_are_capped(tmp_path):
    a = field(180, 360, seed=1)
    toml = '[[layer]]\nid = "small"\ntitle = "Small"\nbody = "earth"\narray = "results/a.npy"\nlevels = [0, 2]\n'
    root = write_project(tmp_path, toml, {"a": a})
    said = []
    manifest = build(root, say=said.append)
    layer = manifest["layers"][0]
    assert layer["grid"]["levels"] == [0, 0]
    assert any("capped at 0" in line for line in said)
    rows = (2 * np.arange(256) + 1) * 180 // 512
    for x in (0, 1):
        cols = (2 * (np.arange(256) + 256 * x) + 1) * 360 // 1024
        _, quantity = read_tile(root / "dist", layer, 0, x, 0)
        assert close(quantity, a[np.ix_(rows, cols)], layer)
    assert not (root / "dist" / "small" / "1").exists()


def test_steps_of_an_axis(tmp_path):
    a = np.stack([field(256, 512, seed=s) for s in range(3)])
    toml = ('[[layer]]\nid = "steps"\ntitle = "Steps"\nbody = "earth"\narray = "results/a.npy"\ntype = "uint16"\n'
            'levels = [0, 0]\naxis = { id = "t", kind = "ordered", unit = "yr", values = [2000, "2010", 2020.5] }\n')
    root = write_project(tmp_path, toml, {"a": a})
    layer = build(root, say=QUIET)["layers"][0]
    assert layer["axes"][0]["values"] == ["2000", "2010", "2020.5"]
    assert layer["files"] == "steps/{t}/{level}/{x}/{y}.png"
    for n, step in enumerate(layer["axes"][0]["values"]):
        _, quantity = read_tile(root / "dist", layer, 0, 1, 0, step=step)
        assert close(quantity, a[n][:, 256:], layer)
    index = json.loads((root / "dist" / "steps" / "index.json").read_text())
    assert index["tiles"] == 6 and index["axes"] == {"t": ["2000", "2010", "2020.5"]} and index["levels"] == [0, 0]
    assert len(index["version"]) == 12 and int(index["version"], 16) >= 0
    assert index["bytes"] == sum(p.stat().st_size for p in (root / "dist" / "steps").rglob("*.png"))


def test_given_min_and_max_are_kept_and_values_outside_are_clipped(tmp_path):
    a = field(256, 512, seed=2)
    toml = ('[[layer]]\nid = "clip"\ntitle = "Clip"\nbody = "earth"\narray = "results/a.npy"\n'
            'min = 0\nmax = 5.1\nlevels = [0, 0]\n')
    root = write_project(tmp_path, toml, {"a": a})
    said = []
    layer = build(root, say=said.append)["layers"][0]
    assert layer["values"]["scale"] == pytest.approx(5.1 / 255) and layer["values"]["offset"] == 0
    assert layer["values"]["nodata"] is None and layer["values"]["unit"] == ""
    assert any("outside min..max" in line for line in said)
    _, quantity = read_tile(root / "dist", layer, 0, 0, 0)
    assert close(quantity, np.clip(a[:, :256], 0, 5.1), layer)


def test_build_is_deterministic(project, tmp_path):
    root, _ = project
    build(root, tmp_path / "one", say=QUIET)
    build(root, tmp_path / "two", say=QUIET)
    files = lambda d: {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}
    one, two = files(tmp_path / "one"), files(tmp_path / "two")
    assert len(one) == 1 + 2 * (1 + 2 + 8)        # the manifest; per layer an index, 2 tiles and 8 tiles
    assert one == two


LAYER = '[[layer]]\nid = "a"\ntitle = "A"\nbody = "earth"\narray = "results/a.npy"\nlevels = [0, 0]\n'


@pytest.mark.parametrize("toml, array, expected", [
    (LAYER, None, "does not exist"),
    (LAYER, np.full((4, 8), np.nan), "only NaN"),
    (LAYER, np.array([[1.0, np.nan], [2.0, 3.0]]), "add `nodata = 255`"),
    (LAYER + "nodata = 7\n", np.ones((4, 8)), "`nodata` must be 0 or 255"),
    (LAYER + 'type = "float32"\n', np.ones((4, 8)), "`type` must be"),
    (LAYER + 'extent = "europe"\n', np.ones((4, 8)), "`extent` must be"),
    (LAYER + "colour = 1\n", np.ones((4, 8)), "unknown key `colour`"),
    (LAYER + "min = 2\nmax = 1\n", np.ones((4, 8)), "min <= max"),
    (LAYER, np.ones((2, 4, 8)), "add `axis`"),
    (LAYER + 'axis = { id = "t", kind = "ordered", unit = "yr", values = ["1"] }\n', np.ones((2, 4, 8)), "one value per step"),
    (LAYER + 'axis = { id = "t", kind = "ordered", unit = "yr", values = ["1"] }\n', np.ones((4, 8)), "remove `axis`"),
    (LAYER.replace("[0, 0]", "[3, 4]"), np.ones((4, 8)), "set levels = [0, 0]"),
    (LAYER.replace('"a"\n', '"../a"\n', 1), np.ones((4, 8)), "id must be a lower-case name"),
    (LAYER.replace('"earth"', '"Earth"'), np.ones((4, 8)), "body must name a sphere"),
    (LAYER + LAYER, np.ones((4, 8)), "same `id`"),
    ("", np.ones((4, 8)), "at least one [[layer]]"),
])
def test_build_refuses_and_says_what_to_change(tmp_path, toml, array, expected):
    root = write_project(tmp_path, toml, {} if array is None else {"a": array})
    with pytest.raises(BuildError) as error:
        build(root, say=QUIET)
    assert expected in str(error.value)
