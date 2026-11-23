"""The atlas's address, in Python: the same cells and the same text as lib/atlas.js (docs/ATLAS.md §2).

A repository that publishes a layer copies this file, as a marble copies marble.js. Every step from a
direction to a path uses only + - * / and the square root, in the order atlas.js uses them, so both
give the same path for the same doubles; tests/atlas.test.mjs runs the two against each other.
"""
import math
import re

SPHERICAL = 30        # the last depth cut on the sphere; below it the cut is flat
FROM_DIRECTION = 48   # the deepest path a direction in doubles can give
MAX_DEPTH = 256
LATLON_DEPTH = 40     # the deepest path given from degrees
_STEP = 52
_MAX_KEY, _MAX_AXES, _MAX_MEMBERS = 4096, 32, 64
EDGE = 1.1071487177940904

_PHI = (1 + math.sqrt(5)) / 2
_RAD = math.pi / 180


def _unit(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


ICO = [_unit(*v) for v in [
    (-1, _PHI, 0), (1, _PHI, 0), (-1, -_PHI, 0), (1, -_PHI, 0),
    (0, -1, _PHI), (0, 1, _PHI), (0, -1, -_PHI), (0, 1, -_PHI),
    (_PHI, 0, -1), (_PHI, 0, 1), (-_PHI, 0, -1), (-_PHI, 0, 1),
]]
RHOMBI = [
    (0, 11, 5, 4), (0, 5, 1, 9), (0, 1, 7, 8), (0, 7, 10, 6), (0, 10, 11, 2),
    (3, 9, 4, 5), (3, 4, 2, 11), (3, 2, 6, 10), (3, 6, 8, 7), (3, 8, 9, 1),
]
_PATH_RE = re.compile(r"[0-9][0-3]*")


def _det(a, b, c):
    # a . ((b - a) x (c - a)): with the differences, see atlas.js
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    return a[0] * (uy * vz - uz * vy) + a[1] * (uz * vx - ux * vz) + a[2] * (ux * vy - uy * vx)


def _mid(a, b):
    return _unit(a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _side(x, y, p, ref):
    return _det(x, y, p) * _det(x, y, ref) >= 0


def valid(path):
    return isinstance(path, str) and bool(_PATH_RE.fullmatch(path)) and len(path) - 1 <= MAX_DEPTH


def depth_of(path):
    return len(path) - 1


def _need(path):
    if not valid(path):
        raise ValueError("not a cell path: %s" % str(path)[:40])


def _children(c):
    A, B, C, D = c
    ab, ac, bc, bd, cd = _mid(A, B), _mid(A, C), _mid(B, C), _mid(B, D), _mid(C, D)
    return [(A, ab, ac, bc), (ab, B, bc, bd), (ac, bc, C, cd), (bc, bd, cd, D)]


def _child_of(c, p):
    A, B, C, D = c
    ab, ac, bc = _mid(A, B), _mid(A, C), _mid(B, C)
    if _side(B, C, p, A):
        if not _side(ab, bc, p, ac):
            return 1, (ab, B, bc, _mid(B, D))
        if not _side(ac, bc, p, ab):
            return 2, (ac, bc, C, _mid(C, D))
        return 0, (A, ab, ac, bc)
    bd, cd = _mid(B, D), _mid(C, D)
    if not _side(bc, bd, p, cd):
        return 1, (ab, B, bc, bd)
    if not _side(bc, cd, p, bd):
        return 2, (ac, bc, C, cd)
    return 3, (bc, bd, cd, D)


def _corners(r):
    return tuple(ICO[v] for v in RHOMBI[r])


def _rhombus_of(p):
    best, best_in = 0, -math.inf
    for r in range(10):
        A, B, C, D = _corners(r)
        for X, Y, Z in ((A, B, C), (B, D, C)):
            s = 1 if _det(X, Y, Z) >= 0 else -1
            inside = min(s * _det(X, Y, p), s * _det(Y, Z, p), s * _det(Z, X, p))
            if inside > best_in:
                best_in, best = inside, r
    return best


def _in_plane(O, P1, P2, p):
    e1 = (P1[0] - O[0], P1[1] - O[1], P1[2] - O[2])
    e2 = (P2[0] - O[0], P2[1] - O[1], P2[2] - O[2])
    n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
    t = (O[0] * n[0] + O[1] * n[1] + O[2] * n[2]) / (p[0] * n[0] + p[1] * n[1] + p[2] * n[2])
    q = (p[0] * t - O[0], p[1] * t - O[1], p[2] * t - O[2])
    d00 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]
    d01 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]
    d11 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]
    d20 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2]
    d21 = q[0] * e2[0] + q[1] * e2[1] + q[2] * e2[2]
    den = d00 * d11 - d01 * d01
    return (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den


def _keep(x):
    return (x if x < 1 else 1 - 2.0 ** -53) if x >= 0 else 0.0


def _flat_uv(c, p):
    # the cell is its two triangles (A, B, C) and (D, C, B): see atlas.js
    A, B, C, D = c
    u, v = _in_plane(A, B, C, p)
    if u + v <= 1:
        return _keep(u), _keep(v)
    s, t = _in_plane(D, C, B, p)
    return _keep(1 - s), _keep(1 - t)


def _flat_point(c, u, v):
    A, B, C, D = c
    O, P1, P2, a, b = (A, B, C, u, v) if u + v <= 1 else (D, C, B, 1 - u, 1 - v)
    return _unit(O[0] + a * (P1[0] - O[0]) + b * (P2[0] - O[0]),
                 O[1] + a * (P1[1] - O[1]) + b * (P2[1] - O[1]),
                 O[2] + a * (P1[2] - O[2]) + b * (P2[2] - O[2]))


def _flat_digits(u, v, n):
    out = []
    for _ in range(n):
        u *= 2
        v *= 2
        a, b = (1 if u >= 1 else 0), (1 if v >= 1 else 0)
        u -= a
        v -= b
        out.append(str(a + 2 * b))
    return "".join(out)


def locate(p, depth):
    """The path of the cell at `depth` that holds direction p (any length but zero)."""
    if not isinstance(depth, int) or depth < 0 or depth > FROM_DIRECTION:
        raise ValueError("depth 0 to %d from a direction" % FROM_DIRECTION)
    n = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
    if not (n > 0) or not math.isfinite(n):
        raise ValueError("not a direction")
    d = (p[0] / n, p[1] / n, p[2] / n)
    r = _rhombus_of(d)
    path = [str(r)]
    c = _corners(r)
    for _ in range(min(depth, SPHERICAL)):
        digit, c = _child_of(c, d)
        path.append(str(digit))
    if depth > SPHERICAL:
        path.append(_flat_digits(*_flat_uv(c, d), depth - SPHERICAL))
    return "".join(path)


def cell(path):
    """A cell from its path: corners to depth 30; below, the depth-30 cell (`anchor`) and (i, j) of 2^levels in it."""
    _need(path)
    depth = depth_of(path)
    c = _corners(int(path[0]))
    for k in range(1, min(depth, SPHERICAL) + 1):
        c = _children(c)[int(path[k])]
    out = {"path": path, "r": int(path[0]), "depth": depth, "corners": c}
    if depth > SPHERICAL:
        i = j = 0
        for ch in path[SPHERICAL + 1:]:
            i, j = 2 * i + (int(ch) & 1), 2 * j + (int(ch) >> 1)
        out.update(anchor=path[:SPHERICAL + 1], i=i, j=j, levels=depth - SPHERICAL)
    return out


def center(path):
    """The direction of a cell's middle (a unit vector)."""
    c = cell(path)
    if c["depth"] <= SPHERICAL:
        return _mid(c["corners"][1], c["corners"][2])
    size = 2.0 ** -c["levels"]
    return _flat_point(c["corners"], _keep((float(c["i"]) + 0.5) * size), _keep((float(c["j"]) + 0.5) * size))


def extend(path, u, v, n):
    """n more digits for the point (u, v) of a cell at depth 30 or below, u and v in [0, 1)."""
    _need(path)
    if depth_of(path) < SPHERICAL:
        raise ValueError("extend() works from depth %d down" % SPHERICAL)
    if not (0 <= u < 1 and 0 <= v < 1):
        raise ValueError("u and v are from 0 up to, not including, 1")
    if not isinstance(n, int) or n < 0 or n > _STEP:
        raise ValueError("0 to %d digits at a time" % _STEP)
    if depth_of(path) + n > MAX_DEPTH:
        raise ValueError("at most depth %d" % MAX_DEPTH)
    return path + _flat_digits(u, v, n)


def lattice(path):
    """(r, depth, i, j) as Lazyverse counts them."""
    _need(path)
    i = j = 0
    for ch in path[1:]:
        i, j = 2 * i + (int(ch) & 1), 2 * j + (int(ch) >> 1)
    return int(path[0]), depth_of(path), i, j


def path_of(r, depth, i, j):
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (r, depth, i, j)):
        raise ValueError("no such cell")
    if not (0 <= r < 10 and 0 <= depth <= MAX_DEPTH and 0 <= i < (1 << depth) and 0 <= j < (1 << depth)):
        raise ValueError("no such cell")
    return str(r) + "".join(str(((i >> k) & 1) + 2 * ((j >> k) & 1)) for k in range(depth - 1, -1, -1))


def dir_of(lat, lon):
    """A direction from degrees: y through the north pole, longitude from x towards z."""
    if lat >= 90:
        return (0.0, 1.0, 0.0)
    if lat <= -90:
        return (0.0, -1.0, 0.0)
    l = math.fmod(lon, 360)
    if l >= 180:
        l -= 360
    elif l < -180:
        l += 360
    a, o = lat * _RAD, l * _RAD
    co = 0.0 if l in (-90, 90) else 1.0 if l == 0 else -1.0 if l == -180 else math.cos(o)
    so = 0.0 if l in (0, -180) else 1.0 if l == 90 else -1.0 if l == -90 else math.sin(o)
    return (math.cos(a) * co, math.sin(a), math.cos(a) * so)


def mercator_lat_lon(z, x, y):
    """The middle of a Web Mercator tile (row y from the north), in degrees."""
    n = 2 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - (2 * (y + 0.5)) / n)))), (x + 0.5) / n * 360 - 180)


_BODY_RE = re.compile(r"[a-z]([a-z0-9-]{0,30}[a-z0-9])?(:[A-Za-z0-9]([A-Za-z0-9._-]{0,62}[A-Za-z0-9])?)?")
_AXIS_RE = re.compile(r"[a-z][a-z0-9-]{0,31}")
_NUMBER_RE = re.compile(r"-?(0|[1-9][0-9]{0,63})(\.[0-9]{1,64})?")
_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,63}")


def _value(v, axis):
    if isinstance(v, (list, tuple, set, frozenset)):
        if not 1 <= len(v) <= _MAX_MEMBERS:
            raise ValueError("axis %s: a set has 1 to %d members" % (axis, _MAX_MEMBERS))
        if any(isinstance(x, (list, tuple, set, frozenset)) for x in v):
            raise ValueError("axis %s: a set of sets" % axis)
        return ",".join(sorted({_value(x, axis) for x in v}))
    if not isinstance(v, str):
        raise TypeError("axis %s: a value is text (a number written out, or a name)" % axis)
    if _NUMBER_RE.fullmatch(v):
        s = v.rstrip("0").rstrip(".") if "." in v else v
        return "0" if re.fullmatch(r"-0(\.0*)?", s) else s
    if _NAME_RE.fullmatch(v):
        return v
    raise ValueError("axis %s: neither a number written out nor a name: %s" % (axis, v[:40]))


def setting_key(body, path, axes=None):
    """The one way to write a setting: body/path?axis=value&axis=value, axes in order of their names."""
    if not isinstance(body, str) or not _BODY_RE.fullmatch(body):
        raise ValueError("not a body")
    _need(path)
    parts = []
    if len(axes or {}) > _MAX_AXES:
        raise ValueError("at most %d axes" % _MAX_AXES)
    for k in sorted(axes or {}):
        if not isinstance(k, str) or not _AXIS_RE.fullmatch(k):
            raise ValueError("not an axis: %s" % k[:40])
        parts.append("%s=%s" % (k, _value(axes[k], k)))
    key = "%s/%s%s" % (body, path, ("?" + "&".join(parts)) if parts else "")
    if len(key) > _MAX_KEY:
        raise ValueError("a setting is at most %d characters" % _MAX_KEY)
    return key


if __name__ == "__main__":
    # for the test: one JSON request per line, one answer per line
    import json
    import sys
    for line in sys.stdin:
        q = json.loads(line)
        if q["op"] == "locate":
            print(locate(q["p"], q["depth"]))
        elif q["op"] == "center":
            print(json.dumps([repr(x) for x in center(q["path"])]))
        elif q["op"] == "key":
            print(setting_key(q["body"], q["path"], q.get("axes")))
