"""check.py: it accepts what build.py writes and the two examples of the specification, and rejects broken manifests."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import atlas
from build import build
from check import check_dist, check_manifest
from conftest import ROOT

QUIET = lambda line: None

RASTER = {
    "id": "people", "title": "People", "icon": "people", "body": "earth", "kind": "raster",
    "focus": {"depth": [0, 12]},
    "values": {"type": "uint16", "scale": 1, "offset": 0, "unit": "residents", "nodata": 65535},
    "grid": {"kind": "latlon", "levels": [0, 3], "tile": 256},
    "axes": [{"id": "t", "kind": "ordered", "unit": "yr", "values": ["2025"]}],
    "files": "people/{t}/{level}/{x}/{y}.png", "index": "people/index.json",
    "maps": ["viridis", "magma"],
}
EVENTS = {
    "id": "gigs", "title": "Gigs", "icon": "work", "body": "earth", "kind": "events",
    "focus": {"depth": [4, 14], "t": ["-2592000", "1209600"]},
    "files": "gigs/{day}.ndjson",
    "app": {"name": "Kittelzone", "home": "https://kittel.zone/", "open": "https://kittel.zone/gig/{id}",
            "join": "https://kittel.zone/join"},
}
GOOD = {"atlas": 1, "source": "weichseltree/nearby", "title": "Nearby", "home": "https://example.org/",
        "licence": "ODbL-1.0", "cite": "GHS-POP 2025, JRC; OpenStreetMap contributors", "layers": [RASTER, EVENTS]}


def change(path, value):
    """A function that sets (or, with value DROP, removes) the entry at `path` of a manifest."""
    def apply(manifest):
        at = manifest
        for key in path[:-1]:
            at = at[key]
        if value is DROP:
            del at[path[-1]]
        else:
            at[path[-1]] = value
    return apply


DROP = object()
L0, L1 = ("layers", 0), ("layers", 1)

BROKEN = [
    ("absolute URL in files", change(L0 + ("files",), "https://evil.example/{t}/{level}/{x}/{y}.png"), "files must be a relative path"),
    ("absolute URL in events files", change(L1 + ("files",), "https://kittel.zone/feed/{day}.ndjson"), "files must be a relative path"),
    ("protocol-relative files", change(L0 + ("files",), "//evil.example/{t}/{level}/{x}/{y}.png"), "files must be a relative path"),
    ("leading slash", change(L0 + ("files",), "/people/{t}/{level}/{x}/{y}.png"), "files must be a relative path"),
    (".. in files", change(L0 + ("files",), "../people/{t}/{level}/{x}/{y}.png"), "files must be a relative path"),
    (".. in index", change(L0 + ("index",), "people/../../index.json"), "index must be a relative path"),
    ("backslash in files", change(L0 + ("files",), "people\\{t}\\{level}\\{x}\\{y}.png"), "files must be a relative path"),
    ("axis missing from files", change(L0 + ("files",), "people/{level}/{x}/{y}.png"), "must contain {t} exactly once"),
    ("unknown placeholder", change(L0 + ("files",), "people/{t}/{level}/{x}/{y}/{z}.png"), "placeholder {z}"),
    ("placeholder in index", change(L0 + ("index",), "people/{level}.json"), "placeholder {level}"),
    ("missing licence", change(("licence",), DROP), "licence: is required"),
    ("empty licence", change(("licence",), "  "), "licence: is required"),
    ("missing cite", change(("cite",), DROP), "cite: is required"),
    ("cite too long", change(("cite",), "x" * 401), "cite: is required"),
    ("duplicate ids", change(L1 + ("id",), "people"), "the id is used twice"),
    ("bad body", change(L0 + ("body",), "Earth"), "body must name a sphere"),
    ("bad body seed", change(L0 + ("body",), "lazyverse:"), "body must name a sphere"),
    ("too many layers", change(("layers",), [dict(RASTER, id="l%d" % n) for n in range(33)]), "layers: must be a list of 1 to 32"),
    ("no layers", change(("layers",), []), "layers: must be a list of 1 to 32"),
    ("atlas as text", change(("atlas",), "1"), "atlas: must be the number 1"),
    ("atlas 2", change(("atlas",), 2), "atlas: must be the number 1"),
    ("atlas true", change(("atlas",), True), "atlas: must be the number 1"),
    ("source without owner", change(("source",), "nearby"), "source: must be owner/repository"),
    ("title a number", change(("title",), 7), "title: must be text"),
    ("title too long", change(("title",), "x" * 81), "title: must be text"),
    ("home over http", change(("home",), "http://example.org/"), "home: must be an https://"),
    ("layers an object", change(("layers",), {"people": RASTER}), "layers: must be a list"),
    ("layer a string", change(L0, "people"), "layer 0: must be a JSON object"),
    ("unknown top key", change(("settings",), []), "unknown key \"settings\""),
    ("unknown layer key", change(L0 + ("default",), {"map": "magma"}), "unknown key \"default\""),
    ("bad id", change(L0 + ("id",), "People"), "id must be a lower-case name"),
    ("bad icon", change(L0 + ("icon",), "<svg>"), "icon must be"),
    ("bad kind", change(L0 + ("kind",), "film"), "kind must be"),
    ("no focus", change(L0 + ("focus",), DROP), "focus must be"),
    ("focus too deep", change(L0 + ("focus", "depth"), [0, 31]), "focus must be"),
    ("focus backwards", change(L0 + ("focus", "depth"), [9, 3]), "focus must be"),
    ("focus as floats", change(L0 + ("focus", "depth"), [0.0, 3.5]), "focus must be"),
    ("focus.t as numbers", change(L1 + ("focus", "t"), [-30, 14]), "focus.t must be"),
    ("focus.t backwards", change(L1 + ("focus", "t"), ["5", "-5"]), "focus.t must be"),
    ("values type", change(L0 + ("values", "type"), "float32"), "values.type must be"),
    ("scale as text", change(L0 + ("values", "scale"), "1"), "values.scale must be a number"),
    ("scale not finite", change(L0 + ("values", "scale"), float("inf")), "values.scale must be a number"),
    ("offset null", change(L0 + ("values", "offset"), None), "values.offset must be a number"),
    ("nodata a float", change(L0 + ("values", "nodata"), 1.5), "values.nodata must be"),
    ("nodata too large", change(L0 + ("values", "nodata"), 65536), "values.nodata must be"),
    ("nodata missing", change(L0 + ("values", "nodata"), DROP), "values must have exactly"),
    ("grid kind", change(L0 + ("grid", "kind"), "mercator"), "grid must be"),
    ("grid tile", change(L0 + ("grid", "tile"), 512), "grid must be"),
    ("grid levels backwards", change(L0 + ("grid", "levels"), [3, 0]), "grid must be"),
    ("too many axes", change(L0 + ("axes",), [dict(RASTER["axes"][0], id="a%d" % n) for n in range(5)]), "at most 4 axes"),
    ("axis kind", change(L0 + ("axes", 0, "kind"), "line"), "kind must be ordered"),
    ("axis value a number", change(L0 + ("axes", 0, "values"), [2025]), "values must be 1 to 1000"),
    ("axis value not a name", change(L0 + ("axes", 0, "values"), ["../x"]), "values must be 1 to 1000"),
    ("axis named x", change(L0 + ("axes", 0, "id"), "x"), "not level, x or y"),
    ("bad map name", change(L0 + ("maps",), ["Viridis"]), "maps must be"),
    ("maps a string", change(L0 + ("maps",), "viridis"), "maps must be"),
    ("role the page's own", change(L0 + ("role",), "light"), "role must be one of"),
    ("over somewhere else", change(L0 + ("over",), "ice"), "over must be one of"),
    ("events with a role", change(L1 + ("role",), "mask"), "unknown key \"role\""),
    ("app over http", change(L1 + ("app", "home"), "http://kittel.zone/"), "app.home must be an https://"),
    ("app open without id", change(L1 + ("app", "open"), "https://kittel.zone/gig"), "app.open must contain {id}"),
    ("app javascript", change(L1 + ("app", "join"), "javascript:alert(1)"), "app.join must be an https://"),
    ("events without day", change(L1 + ("files",), "gigs/all.ndjson"), "must contain {day} exactly once"),
    ("events with values", change(L1 + ("values",), RASTER["values"]), "unknown key \"values\""),
]


def test_the_good_manifest_is_valid():
    assert check_manifest(GOOD) == []
    smallest = {k: v for k, v in GOOD.items() if k != "home"}
    smallest["layers"] = [{k: v for k, v in RASTER.items() if k not in ("icon", "axes", "maps")}]
    smallest["layers"][0]["files"] = "people/{level}/{x}/{y}.png"
    assert check_manifest(smallest) == []


@pytest.mark.parametrize("name, apply, expected", BROKEN, ids=[b[0] for b in BROKEN])
def test_broken_manifests_are_rejected(name, apply, expected):
    manifest = copy.deepcopy(GOOD)
    apply(manifest)
    problems = check_manifest(manifest)
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("wrong", [None, [], "atlas", 1, {}, {"atlas": 1, "layers": None}, {"layers": [None, 3, []]}])
def test_wrong_types_give_problems_and_no_crash(wrong):
    assert check_manifest(wrong)


@pytest.mark.parametrize("body", ["earth", "sky", "lazyverse:seed-1", "planetgame:Save_2.a", "Earth", "earth:", "a-", "moon/2", ""])
def test_bodies_are_judged_as_the_address_library_judges_them(body):
    try:
        atlas.setting_key(body, "0")
        library = True
    except ValueError:
        library = False
    assert (check_manifest(dict(GOOD, layers=[dict(RASTER, body=body)])) == []) == library


def test_the_example_builds_and_validates(tmp_path):
    manifest = build(ROOT, tmp_path / "dist", say=QUIET)
    assert check_manifest(manifest) == []
    assert json.loads((tmp_path / "dist" / "atlas.json").read_text(encoding="utf-8")) == manifest
    problems, summary = check_dist(tmp_path / "dist")
    assert problems == []
    assert "licence: " in summary and "cite: " in summary


def test_check_accepts_the_built_dist_from_the_command_line(project):
    root, _ = project
    build(root, say=QUIET)
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "check.py"), str(root / "dist")], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
    assert "check.py: ok" in done.stdout


def test_check_notices_missing_changed_and_oversized_files(project, monkeypatch):
    root, _ = project
    dist = root / "dist"
    build(root, say=QUIET)
    tile = dist / "uint8" / "1" / "2" / "1.png"
    data = tile.read_bytes()

    tile.unlink()
    problems, _ = check_dist(dist)
    assert any("1 tiles are missing" in p and "uint8/1/2/1.png" in p for p in problems)

    tile.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    problems, _ = check_dist(dist)
    assert any("does not describe the tiles on disk" in p for p in problems)

    tile.write_bytes(b"not a picture, but the same size" + data[32:])
    problems, _ = check_dist(dist)
    assert any("is not a 256 x 256 greyscale PNG" in p for p in problems)

    tile.write_bytes(data)
    assert check_dist(dist)[0] == []
    (dist / "uint8" / "index.json").unlink()
    assert any("index.json is missing" in p for p in check_dist(dist)[0])

    build(root, say=QUIET)
    monkeypatch.setattr("check.MAX_FILE_BYTES", 1000)
    assert any("the limit is 20 MiB per file" in p for p in check_dist(dist)[0])
    monkeypatch.setattr("check.MAX_FILES", 5)
    assert any("the limit" in p and "files" in p for p in check_dist(dist)[0])


def test_check_fails_without_a_manifest(tmp_path):
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "check.py"), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 1
    assert "run tools/build.py first" in done.stdout


def test_events_files_must_be_named_by_day(tmp_path):
    manifest = dict(GOOD, layers=[EVENTS])
    (tmp_path / "gigs").mkdir()
    (tmp_path / "atlas.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "gigs" / "2026-11-23.ndjson").write_text("{}\n", encoding="utf-8")
    assert check_dist(tmp_path)[0] == []
    (tmp_path / "gigs" / "today.ndjson").write_text("{}\n", encoding="utf-8")
    assert any("YYYY-MM-DD" in p for p in check_dist(tmp_path)[0])


def test_atlas_py_is_the_unchanged_copy():
    original = Path.home() / "weichseltree" / "marbles" / "lib" / "atlas.py"
    if not original.is_file():
        pytest.skip("%s is not on this machine" % original)
    assert (ROOT / "tools" / "atlas.py").read_bytes() == original.read_bytes()


@pytest.mark.parametrize("role", ["relief", "mask", "field", "glow"])
def test_a_raster_may_say_its_role_and_where_it_lies(role):
    m = copy.deepcopy(GOOD)
    m["layers"][0]["role"] = role
    m["layers"][0]["over"] = "sea"
    assert check_manifest(m) == []


def _harmonic(waves=None, kind="stops"):
    m = copy.deepcopy(GOOD)
    layer = m["layers"][0]
    layer["axes"] = [{"id": "part", "kind": kind, "unit": "", "values": ["m2c", "m2s", "s2c", "s2s"]}]
    layer["files"] = "tide/{part}/{level}/{x}/{y}.png"
    layer["harmonic"] = {"axis": "part", "waves": waves if waves is not None else [{"name": "M2", "cos": "m2c", "sin": "m2s"}, {"name": "S2", "cos": "s2c", "sin": "s2s"}]}
    return m


def test_a_harmonic_layer_names_its_waves_as_steps_of_its_axis():
    assert check_manifest(_harmonic()) == []
    assert any("harmonic.axis" in p for p in check_manifest(_harmonic(kind="ordered")))
    assert any("name must be one of" in p for p in check_manifest(_harmonic([{"name": "M4", "cos": "m2c", "sin": "m2s"}])))
    assert any("name must be one of" in p for p in check_manifest(_harmonic([{"name": "M2", "cos": "m2c", "sin": "m2s"}] * 2)))
    assert any(".sin must be one of" in p for p in check_manifest(_harmonic([{"name": "M2", "cos": "m2c", "sin": "k1s"}])))
    assert any("harmonic.waves must be" in p for p in check_manifest(_harmonic([])))
