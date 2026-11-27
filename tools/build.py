"""Build dist/ from results/ and layers.toml: quantise each array, cut its lat-lon tile pyramid, write the manifest.

Run `python3 tools/build.py`; the same input always gives the same bytes, and anything inconsistent stops the build.
"""
import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from check import LARGEST, MAX_FILES, check_manifest, fill, layer_problems, measure, tile_count

ROOT = Path(__file__).resolve().parent.parent
TILE = 256
TOP_KEYS = {"source", "title", "home", "licence", "cite", "layer"}
LAYER_KEYS = {"id", "title", "icon", "body", "array", "extent", "unit", "type", "min", "max", "nodata",
              "levels", "focus", "axis", "maps", "role", "over", "harmonic"}


class BuildError(Exception):
    """Something in layers.toml or results/ that the person has to change."""


def need(condition, message):
    if not condition:
        raise BuildError(message)


def known_keys(table, keys, where):
    unknown = sorted(set(table) - keys)
    need(not unknown, "%s: unknown key `%s`; the keys are: %s" % (where, unknown[0] if unknown else "", ", ".join(sorted(keys))))


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_array(root, entry, where):
    """The layer's array as float64 of shape (steps, H, W); NaN marks cells without data."""
    rel = entry.get("array")
    need(isinstance(rel, str) and rel.startswith("results/") and rel.endswith(".npy") and ".." not in rel,
         "%s: set `array` to a .npy file under results/, for example \"results/example/field.npy\"" % where)
    need((root / rel).is_file(), "%s: %s does not exist; write the array there or correct `array`" % (where, rel))
    a = np.load(root / rel, allow_pickle=False)
    need(a.dtype.kind in "fiu", "%s: %s holds %s; save an array of numbers" % (where, rel, a.dtype))
    a = a.astype(np.float64)
    has_axis = "axis" in entry
    if a.ndim == 2:
        need(not has_axis, "%s: %s has shape (H, W) but `axis` is set; remove `axis`, or stack the steps as (T, H, W)" % (where, rel))
        a = a[None]
    else:
        need(a.ndim == 3, "%s: %s has shape %s; it must be (H, W) or (T, H, W)" % (where, rel, a.shape))
        need(has_axis, "%s: %s has shape (T, H, W); add `axis` with one value per step" % (where, rel))
    need(a.shape[1] >= 1 and a.shape[2] >= 1, "%s: %s is empty" % (where, rel))
    need(not np.isinf(a).any(), "%s: %s contains infinity; replace it with NaN (no data) or a number" % (where, rel))
    need(not np.isnan(a).all(), "%s: %s contains only NaN, there is nothing to show" % (where, rel))
    return a


def read_axis(entry, steps, where):
    """The manifest's axis for a (T, H, W) array: its values written as text, one per step."""
    axis = entry["axis"]
    need(isinstance(axis, dict) and set(axis) == {"id", "kind", "unit", "values"},
         "%s: `axis` needs exactly id, kind, unit and values" % where)
    need(isinstance(axis["values"], list) and len(axis["values"]) == steps,
         "%s: `axis.values` must list one value per step; the array has %d steps" % (where, steps))
    values = [v if isinstance(v, str) else repr(v) if is_number(v) else None for v in axis["values"]]
    need(None not in values, "%s: `axis.values` must be numbers or names" % where)
    return {"id": axis["id"], "kind": axis["kind"], "unit": axis["unit"], "values": values}


def value_rule(entry, a, where):
    """The manifest's `values`, and the first and last stored integer that stand for a quantity."""
    kind = entry.get("type", "uint8")
    need(isinstance(kind, str) and kind in LARGEST, "%s: `type` must be \"uint8\" or \"uint16\"" % where)
    largest = LARGEST[kind]
    nodata = entry.get("nodata")
    need(nodata is None or (isinstance(nodata, int) and nodata in (0, largest)),
         "%s: `nodata` must be 0 or %d (the ends of %s), or left out" % (where, largest, kind))
    need(nodata is not None or not np.isnan(a).any(),
         "%s: the array has NaN cells; add `nodata = %d` so they can be stored" % (where, largest))
    lowest = entry.get("min", float(np.nanmin(a)))
    highest = entry.get("max", float(np.nanmax(a)))
    need(is_number(lowest) and is_number(highest) and lowest <= highest,
         "%s: `min` and `max` must be numbers with min <= max" % where)
    first = 1 if nodata == 0 else 0
    last = largest - 1 if nodata == largest else largest
    scale = (highest - lowest) / (last - first) or 1.0
    unit = entry.get("unit", "")
    return {"type": kind, "scale": scale, "offset": lowest - first * scale, "unit": unit, "nodata": nodata}, first, last


def quantise(grid, values, first, last):
    """Quantities to stored integers: round((q - offset) / scale), kept inside first..last; NaN becomes nodata."""
    missing = np.isnan(grid)
    stored = np.rint((np.where(missing, 0.0, grid) - values["offset"]) / values["scale"])
    stored = np.clip(stored, first, last)
    if values["nodata"] is not None:
        stored[missing] = values["nodata"]
    return stored.astype(values["type"])


def top_level(shape):
    """The coarsest level whose grid (256 * 2^L rows, twice as many columns) is at least as fine as the array."""
    level = 0
    while (TILE << level) < shape[0] or (2 * TILE << level) < shape[1]:
        level += 1
    return level


def to_level(a, level):
    """The array on the grid of `level`: every pixel takes the array cell that lies under its middle."""
    h, w = TILE << level, 2 * TILE << level
    rows = (2 * np.arange(h) + 1) * a.shape[0] // (2 * h)
    cols = (2 * np.arange(w) + 1) * a.shape[1] // (2 * w)
    return a[np.ix_(rows, cols)]


def block_mean(a, f):
    """The mean of every f x f block, ignoring NaN; a block of only NaN stays NaN."""
    blocks = a.reshape(a.shape[0] // f, f, a.shape[1] // f, f)
    good = ~np.isnan(blocks)
    count = good.sum(axis=(1, 3))
    total = np.where(good, blocks, 0.0).sum(axis=(1, 3))
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def build_layer(root, dist, entry, say):
    """Write one layer's tiles and index.json into dist/, and return its part of the manifest."""
    need(isinstance(entry, dict), "layers.toml: write every layer as a [[layer]] table")
    where = "layer %s" % entry.get("id", "(without id)")
    known_keys(entry, LAYER_KEYS, where)
    for key in ("id", "title", "body", "array", "levels"):
        need(key in entry, "%s: `%s` is missing in layers.toml" % (where, key))
    need(isinstance(entry["id"], str), "%s: `id` must be text" % where)
    need(entry.get("extent", "global") == "global",
         "%s: `extent` must be \"global\"; this version only takes arrays that cover the whole sphere" % where)
    a = load_array(root, entry, where)
    values, first, last = value_rule(entry, a, where)
    outside = int(((a < entry.get("min", -np.inf)) | (a > entry.get("max", np.inf))).sum())
    if outside:
        say("%s: %d cells lie outside min..max and are stored as min or max" % (where, outside))

    levels = entry["levels"]
    need(isinstance(levels, list) and len(levels) == 2 and all(isinstance(v, int) for v in levels)
         and 0 <= levels[0] <= levels[1], "%s: `levels` must be [lo, hi] with 0 <= lo <= hi" % where)
    top = top_level(a.shape[1:])
    lo, hi = levels
    need(lo <= top, "%s: `levels` starts at %d, but the array (%d x %d) has detail for level %d at most; "
         "set levels = [0, %d]" % (where, lo, a.shape[1], a.shape[2], top, top))
    if hi > top:
        say("%s: levels capped at %d instead of %d: the array is %d x %d, and level %d (%d x %d) already "
            "shows every cell of it" % (where, top, hi, a.shape[1], a.shape[2], top, TILE << top, 2 * TILE << top))
        hi = top

    layer = {"id": entry["id"], "title": entry["title"]}
    if "icon" in entry:
        layer["icon"] = entry["icon"]
    focus = entry.get("focus", [0, 30])
    need(isinstance(focus, list), "%s: `focus` must be [min, max], the depths 0 to 30 at which the layer is offered" % where)
    layer.update(body=entry["body"], kind="raster", focus={"depth": focus}, values=values,
                 grid={"kind": "latlon", "levels": [lo, hi], "tile": TILE})
    steps = "{level}/{x}/{y}.png"
    if "axis" in entry:
        layer["axes"] = [read_axis(entry, len(a), where)]
        steps = "{%s}/%s" % (layer["axes"][0]["id"], steps)
    layer.update(files="%s/%s" % (entry["id"], steps), index="%s/index.json" % entry["id"])
    if "maps" in entry:
        layer["maps"] = entry["maps"]
    # how the page draws it among the other layers that are on, and over which part of the sphere (marbles docs/ATLAS.md §12.4)
    for key in ("role", "over", "harmonic"):
        if key in entry:
            layer[key] = entry[key]
    problems = layer_problems(layer)
    need(not problems, "layers.toml, %s: %s" % (where, "; ".join(problems)))
    need(tile_count(layer) <= MAX_FILES, "%s: levels [%d, %d] make %d tiles, more than the %d files a site may have; "
         "lower the top of `levels`" % (where, lo, hi, tile_count(layer), MAX_FILES))

    for step in range(len(a)):
        names = {layer["axes"][0]["id"]: layer["axes"][0]["values"][step]} if "axes" in layer else {}
        finest = to_level(a[step], top)
        for level in range(lo, hi + 1):
            stored = quantise(block_mean(finest, 1 << (top - level)), values, first, last)
            for x in range(2 << level):
                for y in range(1 << level):
                    tile = stored[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE]
                    path = dist / fill(layer["files"], {**names, "level": level, "x": x, "y": y})
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.ascontiguousarray(tile)).save(path, format="PNG")

    version, tiles, total, _ = measure(dist, layer)
    index = {"version": version, "levels": [lo, hi], "tiles": tiles,
             "axes": {axis["id"]: axis["values"] for axis in layer.get("axes", [])}, "bytes": total}
    write_json(dist / layer["index"], index)
    say("%s: %d tiles, levels %d to %d, %.1f kB, version %s" % (where, tiles, lo, hi, total / 1000, version))
    return layer


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build(root=ROOT, dist=None, say=print):
    """Read root/layers.toml and root/results/, write dist/ (root/dist by default), and return the manifest."""
    root = Path(root)
    dist = Path(dist) if dist else root / "dist"
    need((root / "layers.toml").is_file(), "%s is missing; it lists what to publish" % (root / "layers.toml"))
    try:
        config = tomllib.loads((root / "layers.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise BuildError("layers.toml is not valid TOML: %s" % error)
    known_keys(config, TOP_KEYS, "layers.toml")
    for key in ("source", "title", "licence", "cite"):
        need(key in config, "layers.toml: `%s` is missing; add it at the top" % key)
    entries = config.get("layer")
    need(isinstance(entries, list) and entries, "layers.toml: add at least one [[layer]]")
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    need(len(set(map(str, ids))) == len(ids), "layers.toml: two layers have the same `id`; give every layer its own")

    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    manifest = {"atlas": 1, "source": config["source"], "title": config["title"]}
    if "home" in config:
        manifest["home"] = config["home"]
    manifest.update(licence=config["licence"], cite=config["cite"], layers=[])
    for entry in entries:
        manifest["layers"].append(build_layer(root, dist, entry, say))
    problems = check_manifest(manifest)
    need(not problems, "layers.toml, %s" % "; ".join(problems))
    write_json(dist / "atlas.json", manifest)
    say("wrote %s" % (dist / "atlas.json"))
    return manifest


def main():
    try:
        build()
    except BuildError as error:
        print("build.py: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
