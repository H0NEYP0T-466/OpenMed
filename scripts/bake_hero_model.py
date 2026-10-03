"""Bake a low-poly hero model from the atlas geometry.

The atlas streams 33.6 MB of chunked geometry and takes ~22 s to bind — fine for
an inspection plate, unusable as a front-page fold. This collapses the same
geometry to a single GLB that loads in about a second, by vertex-clustering each
anatomical system onto a shared grid and re-deriving normals from the new faces.

Per-system groups are preserved so the hero keeps the atlas palette.

Run from the repo root:  backend/venv/bin/python scripts/bake_hero_model.py
"""

from __future__ import annotations

import gzip
import json
import re
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "public/models/atlas"
OUT = ATLAS / "hero-body.glb"

# Grid resolution for vertex clustering. Higher = more triangles. Tuned to land
# near 70k triangles, which holds the silhouette at hero size.
GRID = 350

# System colours, read from the same source the atlas uses so the hero cannot
# drift from the plate palette.
def read_system_colours() -> dict[str, tuple[str, str]]:
    src = (ROOT / "src/types/atlas.ts").read_text()
    block = src[src.index("ATLAS_SYSTEMS"):]
    found = re.findall(
        r"id:\s*'([a-z]+)',\s*\n\s*name:\s*'([^']+)',\s*\n\s*color:\s*'(#[0-9a-fA-F]{6})'",
        block,
    )
    return {i: (name, hexcol) for i, name, hexcol in found}


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    srgb = np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])
    # glTF baseColorFactor is linear; the palette is authored in sRGB.
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return tuple(float(v) for v in lin)


# Left out of the front-page model: it is a public-facing fold, and the plate
# itself still carries the full anatomy.
HERO_EXCLUDED_SYSTEMS = frozenset({"reproductive"})


def load_system_meshes(meta: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decode every part into per-system (positions, indices) float/int arrays."""
    by_chunk: dict[int, list[dict]] = {}
    for p in meta["parts"]:
        if p["system"] in HERO_EXCLUDED_SYSTEMS:
            continue
        by_chunk.setdefault(p["chunk"], []).append(p)

    acc: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ci, chunkspec in enumerate(meta["chunks"]):
        if ci not in by_chunk:
            continue
        raw = gzip.decompress((ROOT / "public" / chunkspec["gzip"].lstrip("/")).read_bytes())
        for p in by_chunk[ci]:
            pos = np.frombuffer(
                raw, dtype="<f4", count=p["vertexCount"] * 3, offset=p["positions"]
            ).reshape(-1, 3)
            idx = np.frombuffer(
                raw, dtype="<u4", count=p["indexCount"], offset=p["indices"]
            )
            acc.setdefault(p["system"], []).append((pos.astype(np.float32), idx))
    return {s: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v]))
            for s, v in acc.items()}


def cluster_decimate(
    pos: np.ndarray, idx: np.ndarray, origin: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: snap vertices to a grid, then drop what collapses."""
    cells = np.floor((pos - origin) / cell).astype(np.int64) + 512
    key = (cells[:, 0] * 1024 + cells[:, 1]) * 1024 + cells[:, 2]

    _, inv = np.unique(key, return_inverse=True)
    inv = inv.astype(np.int64)
    n = int(inv.max()) + 1

    # Representative = centroid of the cell's members, which keeps the surface
    # closer to the original than snapping to the cell centre.
    new_pos = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(new_pos, inv, pos)
    np.add.at(counts, inv, 1.0)
    new_pos /= counts[:, None]

    tri = inv[idx].reshape(-1, 3)
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[keep]

    # Collapsing merges many triangles onto the same three vertices; keep one.
    _, first = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
    tri = tri[np.sort(first)]

    return new_pos.astype(np.float32), tri.astype(np.uint32)


def compute_normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from the decimated faces."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    face = np.cross(b - a, c - a)  # magnitude is 2x area, so this weights by area
    nrm = np.zeros_like(pos, dtype=np.float64)
    for k in range(3):
        np.add.at(nrm, tri[:, k], face)
    lens = np.linalg.norm(nrm, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return (nrm / lens).astype(np.float32)


def write_glb(path: Path, prims: list[dict]) -> None:
    """Minimal glTF 2.0 binary writer — one mesh, one primitive per system."""
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    mesh_prims: list[dict] = []

    def add_view(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    for prim in prims:
        pos, nrm, tri = prim["positions"], prim["normals"], prim["indices"]
        vmin = pos.min(axis=0).astype(float).tolist()
        vmax = pos.max(axis=0).astype(float).tolist()

        pv = add_view(pos.astype("<f4").tobytes(), 34962)
        accessors.append({"bufferView": pv, "componentType": 5126, "count": int(pos.shape[0]),
                          "type": "VEC3", "min": vmin, "max": vmax})
        a_pos = len(accessors) - 1

        nv = add_view(nrm.astype("<f4").tobytes(), 34962)
        accessors.append({"bufferView": nv, "componentType": 5126, "count": int(nrm.shape[0]),
                          "type": "VEC3"})
        a_nrm = len(accessors) - 1

        iv = add_view(tri.astype("<u4").tobytes(), 34963)
        accessors.append({"bufferView": iv, "componentType": 5125, "count": int(tri.size),
                          "type": "SCALAR"})
        a_idx = len(accessors) - 1

        name, hexcol = prim["name"], prim["color"]
        materials.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*hex_to_rgb(hexcol), 1.0],
                "metallicFactor": 0.08,
                "roughnessFactor": 0.62,
            },
        })
        mesh_prims.append({
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm},
            "indices": a_idx,
            "material": len(materials) - 1,
        })

    gltf = {
        "asset": {"version": "2.0", "generator": "OpenMed hero bake"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "HumanBody"}],
        "meshes": [{"name": "HumanBody", "primitives": mesh_prims}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bn = bytes(blob) + b"\x00" * ((4 - len(blob) % 4) % 4)

    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bn))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(bn), 0x004E4942) + bn
    path.write_bytes(bytes(out))


def main() -> None:
    meta = json.loads((ATLAS / "atlas.json").read_text())
    colours = read_system_colours()
    print(f"systems in palette: {len(colours)}")

    meshes = load_system_meshes(meta)
    total_in = sum(len(i) // 3 for _, i in meshes.values())
    print(f"decoded {total_in:,} triangles across {len(meshes)} systems")

    # One grid for every system, so neighbouring systems stay aligned.
    allpos = np.concatenate([p for p, _ in meshes.values()])
    lo, hi = allpos.min(axis=0), allpos.max(axis=0)
    cell = float((hi - lo).max()) / GRID
    print(f"grid {GRID} cells, cell size {cell:.5f}")

    prims = []
    for sid, (pos, idx) in meshes.items():
        dp, dt = cluster_decimate(pos, idx, lo, cell)
        dn = compute_normals(dp, dt)
        name, hexcol = colours.get(sid, (sid.title(), "#b0b0b0"))
        prims.append({"positions": dp, "normals": dn, "indices": dt, "name": name, "color": hexcol})
        print(f"  {sid:15s} {len(idx)//3:8,} -> {len(dt):7,} tris   {len(dp):7,} verts   {hexcol}")

    prims.sort(key=lambda p: p["name"])
    total_out = sum(len(p["indices"]) for p in prims)
    print(f"\ntotal {total_in:,} -> {total_out:,} triangles "
          f"({total_out/total_in*100:.1f}%)")

    write_glb(OUT, prims)
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
