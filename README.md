<!-- tools/atlas.py is an unchanged copy of lib/atlas.py in weichseltree/marbles (docs/ATLAS.md, section 2). Change it there, never here. -->
# atlas-layer

A template for a repository that publishes map layers to the atlas on weichseltree.com.

You put arrays into `results/`. A GitHub Action turns them into small image tiles and one
manifest, `atlas.json`, and publishes both on this repository's GitHub Pages site. The page reads
the manifest when someone opens it, and draws the numbers. No code from this repository ever runs
on the page: it ships numbers and one manifest.

## Three steps

1. **Put arrays into `results/`.** One `.npy` file per layer, shape `(H, W)`, or `(T, H, W)` for
   `T` steps along one axis (years, for example). The array covers the whole sphere: row 0 is the
   north, column 0 is at 180° west. Cells without data are `NaN`.
2. **List them in `layers.toml`.** The file in this template is the example, and every line in it
   is explained there. Set `source`, `title`, `licence` and `cite` at the top, then one `[[layer]]`
   per array. A layer may say how the page draws it among the other layers that are on (`role`:
   `relief` for heights, `mask` for land and sea, `field` for a colour map, `glow` for light such as
   people at night; none is a field) and where a field lies (`over`: `all`, `sea` or `land`).
   A tide model's waves can be one `harmonic` layer: each wave's A cos G and A sin G are two steps
   of the layer's `axis` (kind `stops`), named in `harmonic = { axis = "...", waves = [{ name = "M2",
   cos = "...", sin = "..." }, ...] }`; the page sums them for the time it shows (waves M2, S2, N2,
   K2, K1, O1, P1, Q1, each once, every step its own; `body = "earth"` only, and `levels` no higher
   than 2, since the page sums them on its coarse sphere). weichseltree/atlas-tides is one.
3. **Push to `main`.** The Action builds, checks, tests and publishes. Once, before the first push:
   in the repository's Settings, under Pages, set the source to "GitHub Actions".

To start, click "Use this template" on GitHub, then delete `results/example/` and the two example
layers when you have your own.

## Run it on your own machine

You need Python 3.11 or newer with numpy, Pillow and pytest:

```
python3 -m venv .venv && .venv/bin/pip install numpy pillow pytest
.venv/bin/python tools/build.py      # results/ + layers.toml -> dist/
.venv/bin/python tools/check.py      # is dist/ fit to publish?
.venv/bin/python -m pytest -q        # the tests of the tools
.venv/bin/python tools/make_example.py   # only to write the example arrays again
```

`build.py` stops with a sentence that says what to change. `check.py` prints one line per problem
and exits with an error; then nothing is published.

## What gets published

Everything in `dist/`, and nothing else:

- `atlas.json`: the manifest. Who publishes, the licence, the citation, and for every layer where
  its files are and what a stored number means.
- `<layer>/index.json`: the layer's version (a hash of all its tiles), its levels, the number of
  tiles and their size.
- `<layer>/{level}/{x}/{y}.png`: the tiles, 256 × 256 pixels, greyscale, 8 or 16 bit. With an axis
  the path is `<layer>/{step}/{level}/{x}/{y}.png`, the step written as in `layers.toml`.

How the numbers are stored:

- A stored value `v` means `v * scale + offset` in the layer's unit. `nodata` is one stored value
  that means "nothing here".
- Level `L` has 2^(L+1) × 2^L tiles. Level 0 is the whole sphere in two tiles.
- The top level is the first one that shows every cell of your array. Finer levels would only
  repeat pixels, so `build.py` does not make them, and says so. Each pixel of the top level takes
  the array cell under its middle. Coarser levels are means of blocks of it, ignoring cells
  without data.
- Best is an array of 256·2^k × 512·2^k cells (512 × 1024, 1024 × 2048, …): then the top level is
  your array, cell for cell.
- The same input gives the same bytes on the same machine. Another version of the PNG library may
  compress differently; then the pixels are the same and the layer's version changes.

Limits, checked before publishing: no file over 20 MiB, at most 15 000 files. That is a full
pyramid up to level 6 for one layer. Larger layers need another home than a Pages site.

## Licence and citation are a duty

`licence` and `cite` in `layers.toml` are required, and the page shows them next to the layer.
They are about your **data**: write the licence the data is under, and name everyone its sources
ask you to name. If you may not pass the data on, do not put it here. The `LICENSE` file (MIT) is
only about the template's own code.

## It is public

`dist/` is public as soon as the Action has published it, for anyone who knows the address. The
moment the page reads the manifest, every visitor sees it. If the repository is public, so is
`results/`. Publish nothing about a person, and nothing you would take back.

## Reading it from another site

The page lives at another address than these files, so this site must allow other sites to read
them. GitHub Pages sends
`access-control-allow-origin: *` on every file (seen on this template's own site on 2026-11-24).
To see it for yours:

```
curl -sI https://<owner>.github.io/<repository>/atlas.json | grep -i access-control
```

## What is in this template

```
layers.toml                  what you publish: you edit this
results/                     your arrays (results/example/ is the example)
tools/build.py               results/ + layers.toml -> dist/
tools/check.py               the manifest against the rules of version 1, the files, the sizes
tools/atlas.py               the atlas's address library, for your own scripts (a copy)
tools/make_example.py        writes the example arrays
tests/                       a tile read back equals the array it was cut from, and more
.github/workflows/site.yml   on push: build, check, test, publish
```

Version 1 of the template builds raster layers on a global latitude-longitude grid. `check.py`
already knows the second kind of layer, `events` (a feed of dated items as `{day}.ndjson`), but
`build.py` does not write one yet.
