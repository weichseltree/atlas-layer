"""Check dist/ before it is published: the manifest against the rules of version 1, and every file it names.

Run `python3 tools/check.py`; it prints a short summary, and one line per problem, and exits non-zero on any.
"""
import hashlib
import itertools
import json
import math
import re
import struct
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 15000
LARGEST = {"uint8": 255, "uint16": 65535}

SOURCE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
ID = re.compile(r"[a-z][a-z0-9-]{0,31}")
BODY = re.compile(r"[a-z]([a-z0-9-]{0,30}[a-z0-9])?(:[A-Za-z0-9]([A-Za-z0-9._-]{0,62}[A-Za-z0-9])?)?")
NUMBER = re.compile(r"-?(0|[1-9][0-9]{0,63})(\.[0-9]{1,64})?")
NAME = re.compile(r"[a-z][a-z0-9-]{0,63}")
# a host as it is written in the address bar: lower-case names with dots, the last one not a number
# (no machine addresses, no "localhost"), no dot at the end, no port: the same rule as the page's
HTTPS = re.compile(r"https://([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,61}[a-z0-9](/[^\s\\]*)?")
PATH = re.compile(r"[A-Za-z0-9._/{}-]+")
PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

TOP_KEYS = {"atlas", "source", "title", "home", "licence", "cite", "layers"}
RASTER_KEYS = {"id", "title", "icon", "body", "kind", "focus", "values", "grid", "axes", "files", "index", "maps"}
EVENTS_KEYS = {"id", "title", "icon", "body", "kind", "focus", "files", "app"}


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_text(v, shortest, longest, pattern=None):
    return isinstance(v, str) and shortest <= len(v) <= longest and (pattern is None or bool(pattern.fullmatch(v)))


def is_https(v):
    return is_text(v, 1, 400, HTTPS)


def is_int_pair(v, lowest, highest):
    return (isinstance(v, list) and len(v) == 2 and all(is_int(x) for x in v)
            and lowest <= v[0] <= v[1] <= highest)


def path_problem(path, placeholders):
    """Why `path` is not a relative path inside dist/ with exactly these {placeholders}, or None."""
    if not is_text(path, 1, 200, PATH) or path.startswith("/") or path.endswith("/") or ".." in path or "//" in path:
        return ("must be a relative path inside dist/ (no https://, no leading slash, no \"..\", no backslash), "
                "written with letters, digits, . _ - / and {placeholders} only")
    found = PLACEHOLDER.findall(path)
    if path.count("{") != len(found) or path.count("}") != len(found):
        return "has a brace that does not belong to a {placeholder}"
    for name in found:
        if name not in placeholders:
            return "has the placeholder {%s}; allowed here: %s" % (name, " ".join("{%s}" % p for p in placeholders) or "none")
    for name in placeholders:
        if found.count(name) != 1:
            return "must contain {%s} exactly once" % name
    return None


def fill(template, names):
    """A path template with its {placeholders} replaced by the values in `names`."""
    for name, value in names.items():
        template = template.replace("{%s}" % name, str(value))
    return template


def tile_paths(layer):
    """The path of every tile of a raster layer, in the fixed order the version hash is taken in:
    axis steps in the manifest's order, then level, then x, then y."""
    axes = layer.get("axes", [])
    ids = [axis["id"] for axis in axes]
    lo, hi = layer["grid"]["levels"]
    for steps in itertools.product(*[axis["values"] for axis in axes]):
        for level in range(lo, hi + 1):
            for x in range(2 << level):
                for y in range(1 << level):
                    yield fill(layer["files"], {**dict(zip(ids, steps)), "level": level, "x": x, "y": y})


def tile_count(layer):
    """How many tiles a raster layer has: level L is 2^(L+1) x 2^L tiles, once per combination of axis steps."""
    lo, hi = layer["grid"]["levels"]
    return math.prod(len(axis["values"]) for axis in layer.get("axes", [])) * sum(2 << (2 * level) for level in range(lo, hi + 1))


def measure(dist, layer):
    """(version, tiles, bytes) of a raster layer's tiles on disk, plus the paths that are missing."""
    digest, count, total, missing = hashlib.sha256(), 0, 0, []
    for rel in tile_paths(layer):
        path = Path(dist) / rel
        if not path.is_file():
            missing.append(rel)
            continue
        data = path.read_bytes()
        digest.update(data)
        count += 1
        total += len(data)
    return digest.hexdigest()[:12], count, total, missing


def check_manifest(obj):
    """Every way `obj` breaks the rules of manifest version 1, one sentence each; an empty list means valid."""
    problems = []
    if not isinstance(obj, dict):
        return ["the manifest must be a JSON object"]
    for key in sorted(set(obj) - TOP_KEYS):
        problems.append("unknown key \"%s\": remove it (version 1 knows %s)" % (key, ", ".join(sorted(TOP_KEYS))))
    if not (is_int(obj.get("atlas")) and obj["atlas"] == 1):
        problems.append("atlas: must be the number 1")
    if not is_text(obj.get("source"), 1, 200, SOURCE):
        problems.append("source: must be owner/repository, for example weichseltree/my-project")
    if not is_text(obj.get("title"), 1, 80):
        problems.append("title: must be text of 1 to 80 characters")
    if "home" in obj and not is_https(obj["home"]):
        problems.append("home: must be an https:// address (or leave it out)")
    for key in ("licence", "cite"):
        if not (is_text(obj.get(key), 1, 400) and obj[key].strip()):
            problems.append("%s: is required; give 1 to 400 characters" % key)
    layers = obj.get("layers")
    if not (isinstance(layers, list) and 1 <= len(layers) <= 32):
        problems.append("layers: must be a list of 1 to 32 layers")
        return problems
    seen = set()
    for n, layer in enumerate(layers):
        if not isinstance(layer, dict):
            problems.append("layer %d: must be a JSON object" % n)
            continue
        where = "layer %s" % layer["id"] if is_text(layer.get("id"), 1, 32, ID) else "layer %d" % n
        for problem in layer_problems(layer):
            problems.append("%s: %s" % (where, problem))
        if isinstance(layer.get("id"), str):
            if layer["id"] in seen:
                problems.append("%s: the id is used twice; give every layer its own id" % where)
            seen.add(layer["id"])
    return problems


def layer_problems(layer):
    """Every way one layer breaks the rules, without the layer's name in front."""
    out = []
    if not is_text(layer.get("id"), 1, 32, ID):
        out.append("id must be a lower-case name like \"people\" (a letter, then letters, digits or -, at most 32)")
    if not is_text(layer.get("title"), 1, 80):
        out.append("title must be text of 1 to 80 characters")
    if "icon" in layer and not is_text(layer["icon"], 1, 32, ID):
        out.append("icon must be a lower-case name like \"people\" (or leave it out)")
    if not is_text(layer.get("body"), 1, 96, BODY):
        out.append("body must name a sphere, for example \"earth\" or \"lazyverse:seed-1\"")
    focus = layer.get("focus")
    if not isinstance(focus, dict) or set(focus) - {"depth", "t"} or not is_int_pair(focus.get("depth"), 0, 30):
        out.append("focus must be {\"depth\": [min, max]} with whole numbers 0 to 30, min <= max, and optionally \"t\"")
    elif "t" in focus:
        t = focus["t"]
        if not (isinstance(t, list) and len(t) == 2 and all(is_text(x, 1, 130, NUMBER) for x in t)
                and Decimal(t[0]) <= Decimal(t[1])):
            out.append("focus.t must be [from, to], two decimal numbers written as text, from <= to")
    kind = layer.get("kind")
    if kind == "raster":
        out += _raster_problems(layer)
    elif kind == "events":
        out += _events_problems(layer)
    else:
        out.append("kind must be \"raster\" or \"events\"")
    return out


def _raster_problems(layer):
    out = ["unknown key \"%s\" for a raster layer: remove it" % key for key in sorted(set(layer) - RASTER_KEYS)]
    values = layer.get("values")
    if not isinstance(values, dict) or set(values) != {"type", "scale", "offset", "unit", "nodata"}:
        out.append("values must have exactly type, scale, offset, unit and nodata")
    else:
        largest = LARGEST.get(values["type"]) if isinstance(values["type"], str) else None
        if largest is None:
            out.append("values.type must be \"uint8\" or \"uint16\"")
        if not is_number(values["scale"]) or values["scale"] == 0:
            out.append("values.scale must be a number, and not 0")
        if not is_number(values["offset"]):
            out.append("values.offset must be a number")
        if not is_text(values["unit"], 0, 16):
            out.append("values.unit must be text of at most 16 characters (it may be empty)")
        nodata = values["nodata"]
        if nodata is not None and not (is_int(nodata) and 0 <= nodata <= (largest or 65535)):
            out.append("values.nodata must be null or a whole number the type can hold")
    grid = layer.get("grid")
    grid_ok = (isinstance(grid, dict) and set(grid) == {"kind", "levels", "tile"} and grid["kind"] == "latlon"
               and is_int_pair(grid["levels"], 0, 12) and is_int(grid["tile"]) and grid["tile"] == 256)
    if not grid_ok:
        out.append("grid must be {\"kind\": \"latlon\", \"levels\": [lo, hi], \"tile\": 256} with 0 <= lo <= hi <= 12")
    axis_ids = []
    axes = layer.get("axes", [])
    if not (isinstance(axes, list) and len(axes) <= 4):
        out.append("axes must be a list of at most 4 axes (or leave it out)")
        axes = []
    for n, axis in enumerate(axes):
        if not isinstance(axis, dict) or set(axis) != {"id", "kind", "unit", "values"}:
            out.append("axis %d must have exactly id, kind, unit and values" % n)
            continue
        if not is_text(axis["id"], 1, 32, ID) or axis["id"] in ("level", "x", "y") or axis["id"] in axis_ids:
            out.append("axis %d: id must be a lower-case name of its own, and not level, x or y" % n)
        else:
            axis_ids.append(axis["id"])
        if axis["kind"] not in ("ordered", "cyclic", "stops", "set"):
            out.append("axis %d: kind must be ordered, cyclic, stops or set" % n)
        if not is_text(axis["unit"], 0, 16):
            out.append("axis %d: unit must be text of at most 16 characters (it may be empty)" % n)
        steps = axis["values"]
        if not (isinstance(steps, list) and 1 <= len(steps) <= 1000
                and all(is_text(s, 1, 130, NUMBER) or is_text(s, 1, 64, NAME) for s in steps)
                and len(set(steps)) == len(steps)):
            out.append("axis %d: values must be 1 to 1000 different entries, each a number written as text "
                       "(\"2025\") or a lower-case name" % n)
    if len(axis_ids) == len(axes):
        problem = path_problem(layer.get("files"), ["level", "x", "y"] + axis_ids)
        if problem:
            out.append("files %s" % problem)
    problem = path_problem(layer.get("index"), [])
    if problem:
        out.append("index %s" % problem)
    if "maps" in layer:
        maps = layer["maps"]
        if not (isinstance(maps, list) and 1 <= len(maps) <= 16 and all(is_text(m, 1, 32, ID) for m in maps)
                and len(set(maps)) == len(maps)):
            out.append("maps must be 1 to 16 different colour map names like \"viridis\" (or leave it out)")
    return out


def _events_problems(layer):
    out = ["unknown key \"%s\" for an events layer: remove it" % key for key in sorted(set(layer) - EVENTS_KEYS)]
    problem = path_problem(layer.get("files"), ["day"])
    if problem:
        out.append("files %s" % problem)
    if "app" in layer:
        app = layer["app"]
        if not isinstance(app, dict) or set(app) != {"name", "home", "open", "join"}:
            out.append("app must have exactly name, home, open and join (or leave app out)")
        else:
            if not is_text(app["name"], 1, 40):
                out.append("app.name must be text of 1 to 40 characters")
            for key in ("home", "open", "join"):
                if not is_https(app[key]):
                    out.append("app.%s must be an https:// address" % key)
            if is_https(app["open"]) and "{id}" not in app["open"]:
                out.append("app.open must contain {id}, the place of the item's id")
    return out


def _raster_file_problems(dist, layer):
    out = []
    index_path = dist / layer["index"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["%s is missing or not JSON; run tools/build.py again" % layer["index"]]
    if tile_count(layer) > MAX_FILES:
        return ["has %d tiles, the limit for dist/ is %d files; lower the top of `levels` in layers.toml"
                % (tile_count(layer), MAX_FILES)]
    version, tiles, total, missing = measure(dist, layer)
    if missing:
        out.append("%d tiles are missing, the first is %s; run tools/build.py again" % (len(missing), missing[0]))
    expected = {"version": version, "levels": layer["grid"]["levels"], "tiles": tiles,
                "axes": {axis["id"]: axis["values"] for axis in layer.get("axes", [])}, "bytes": total}
    if not missing and index != expected:
        different = sorted(k for k in set(index) | set(expected) if index.get(k) != expected.get(k))
        out.append("%s does not describe the tiles on disk (%s differ); run tools/build.py again"
                   % (layer["index"], ", ".join(different)))
    depth = 8 if layer["values"]["type"] == "uint8" else 16
    for rel in tile_paths(layer):
        path = dist / rel
        if path.is_file():
            with open(path, "rb") as f:
                head = f.read(26)
            # a PNG starts with 8 fixed bytes, then the IHDR chunk: width, height, bit depth, colour type (0 = grey)
            if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">IIBB", head[16:26]) != (256, 256, depth, 0):
                out.append("%s is not a 256 x 256 greyscale PNG of %d bit; run tools/build.py again" % (rel, depth))
                break
    return out


def _events_file_problems(dist, layer):
    before, after = layer["files"].split("{day}")
    out = []
    for path in sorted(dist.glob(fill(layer["files"], {"day": "*"}))):
        rel = path.relative_to(dist).as_posix()
        day = rel[len(before):len(rel) - len(after)]
        if not DAY.fullmatch(day):
            out.append("%s: the day in the name must be written YYYY-MM-DD" % rel)
    return out


def check_dist(dist):
    """(problems, summary) for a built dist/ directory."""
    dist = Path(dist)
    try:
        manifest = json.loads((dist / "atlas.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["%s is missing or not JSON; run tools/build.py first" % (dist / "atlas.json")], ""
    problems = check_manifest(manifest)
    if problems:
        return problems, ""
    for layer in manifest["layers"]:
        check = _raster_file_problems if layer["kind"] == "raster" else _events_file_problems
        problems += ["layer %s: %s" % (layer["id"], p) for p in check(dist, layer)]
    files = sorted(p for p in dist.rglob("*") if p.is_file())
    sizes = [p.stat().st_size for p in files]
    if len(files) > MAX_FILES:
        problems.append("dist/ has %d files, the limit is %d; lower the top of `levels` in layers.toml" % (len(files), MAX_FILES))
    for path, size in zip(files, sizes):
        if size > MAX_FILE_BYTES:
            problems.append("%s is %.1f MiB, the limit is 20 MiB per file" % (path.relative_to(dist).as_posix(), size / 2 ** 20))
    summary = "%s: %d layer(s) (%s), %d files, %.1f kB\nlicence: %s\ncite: %s" % (
        manifest["source"], len(manifest["layers"]), ", ".join(layer["id"] for layer in manifest["layers"]),
        len(files), sum(sizes) / 1000, manifest["licence"], manifest["cite"])
    return problems, summary


def main():
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
    problems, summary = check_dist(dist)
    if summary:
        print(summary)
    for problem in problems:
        print("PROBLEM: %s" % problem)
    print("check.py: %s" % ("%d problem(s), nothing should be published" % len(problems) if problems else "ok"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
