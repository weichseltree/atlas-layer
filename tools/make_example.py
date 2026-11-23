"""Write the two small example arrays in results/example/ that let the template build out of the box.

The fields are made from sines and cosines only, so every run writes the same numbers; delete them once you have your own.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "example"


def field(rows, phase=0.0):
    """A smooth field on a global grid of `rows` x 2 * `rows` cells: row 0 is north, column 0 is at 180 W."""
    lat = np.radians(90 - (np.arange(rows) + 0.5) * 180 / rows)[:, None]
    lon = np.radians(-180 + (np.arange(2 * rows) + 0.5) * 180 / rows)[None, :]
    return np.cos(lat) * np.sin(3 * lon + phase) + 0.5 * np.sin(2 * lat)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    one = field(180)
    one[170:, :] = np.nan          # no data south of 80 S, to show how missing cells are stored
    np.save(OUT / "field.npy", one.astype(np.float16))
    steps = np.stack([field(90, phase) for phase in (0.0, 1.0, 2.0)])
    np.save(OUT / "steps.npy", steps.astype(np.float16))
    print("wrote %s: field.npy %s, steps.npy %s" % (OUT, one.shape, steps.shape))


if __name__ == "__main__":
    main()
