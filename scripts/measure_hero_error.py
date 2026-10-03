"""Measure how far the hero decimation departs from the full-resolution body.

"Does it look lumpy?" is a question about surface error. This samples both
surfaces and reports the distance between them, expressed as a percentage of the
body's height so the number means something.

Also sweeps the clustering grid, so the quality/size trade-off is a table rather
than a guess.

Run from the repo root:  backend/venv/bin/python scripts/measure_hero_error.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from bake_hero_model import (  # noqa: E402
    ATLAS,
    cluster_decimate,
    load_system_meshes,
)

SAMPLE_TARGET = 700_000
GRIDS = (110, 150, 200, 260, 320)


def sample_surface(pos: np.ndarray, tri: np.ndarray, n: int, rng) -> np.ndarray:
    """Area-weighted point sample of a triangle mesh."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if total == 0:
        return pos
    p = areas / total
    picks = rng.choice(len(tri), size=n, p=p)

    t0, t1, t2 = tri[picks, 0], tri[picks, 1], tri[picks, 2]
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return (1 - u - v)[:, None] * pos[t0] + u[:, None] * pos[t1] + v[:, None] * pos[t2]


def main() -> None:
    meta = __import__("json").loads((ATLAS / "atlas.json").read_text())
    meshes = load_system_meshes(meta)

    rng = np.random.default_rng(11)

    print("sampling the full-resolution surface…")
    orig_pos, orig_tri = [], []
    offset = 0
    for pos, tri in meshes.values():
        orig_pos.append(pos)
        orig_tri.append(tri + offset)
        offset += len(pos)
    orig_pos = np.concatenate(orig_pos)
    # load_system_meshes yields flat index arrays; sample_surface wants (N, 3).
    orig_tri = np.concatenate(orig_tri).reshape(-1, 3)

    lo = orig_pos.min(axis=0)
    height = float((orig_pos.max(axis=0) - lo).max())

    # Direction matters. Querying decimated points against the original only
    # asks "do the new vertices lie on the old surface?" — they do by
    # construction, so that metric saturates and cannot see lost detail.
    # The informative direction is original -> decimated: how far does the real
    # surface have to travel to reach what we kept.
    ref = sample_surface(orig_pos, orig_tri, SAMPLE_TARGET, rng)
    print(f"reference: {len(ref):,} points · body extent {height:.3f} units\n")

    print(f"{'grid':>6} {'tris':>10} {'reduction':>10} {'RMS err':>9} {'p99 err':>9} "
          f"{'max err':>9} {'RMS %H':>8} {'est. MB':>8}")
    print("-" * 78)

    total_orig = len(orig_tri)

    for grid in GRIDS:
        cell = height / grid
        d_pos, d_tri = [], []
        off = 0
        for pos, tri in meshes.values():
            dp, dt = cluster_decimate(pos, tri, lo, cell)
            d_pos.append(dp)
            d_tri.append(dt + off)
            off += len(dp)
        dp = np.concatenate(d_pos)
        dt = np.concatenate(d_tri)

        # Dense sample of what we kept, then ask how far the original surface
        # is from it. More sample points than triangles so the query reflects
        # the surface rather than the vertex spacing.
        samp = sample_surface(dp, dt, SAMPLE_TARGET, rng)
        d, _ = cKDTree(samp).query(ref, k=1, workers=-1)

        rms = float(np.sqrt((d ** 2).mean()))
        p99 = float(np.percentile(d, 99))
        mx = float(d.max())
        # Rough GLB size: 12 B/vertex for position+normal, 12 B/triangle indices.
        mb = (len(dp) * 24 + len(dt) * 12) / 1e6

        print(f"{grid:>6} {len(dt):>10,} {len(dt)/total_orig*100:>9.1f}% {rms:>9.5f} "
              f"{p99:>9.5f} {mx:>9.5f} {rms/height*100:>7.2f}% {mb:>8.2f}")

    print(f"\nMeasured original -> decimated: the typical distance the real surface")
    print(f"has to travel to reach the kept geometry. Against a body height of")
    print(f"{height:.3f} units. Point-to-point, so it slightly overestimates true")
    print("surface distance — the ranking between grids is what matters.")


if __name__ == "__main__":
    main()
