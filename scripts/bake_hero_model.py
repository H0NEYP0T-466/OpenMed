"""Bake a clean, lightweight hero model from the atlas geometry.

The atlas streams 33.6 MB across 15 chunks (2,234 parts) which takes ~22 s to bind.
This script merges all parts cleanly per anatomical system (with author normals and
proper index offsets), writes a single GLB, and optionally optimizes via gltfpack.

The integumentary (skin) system is configured with alphaMode BLEND (translucent),
allowing the muscular, skeletal, and vascular systems to show through just like
the atlas viewer, but in a single fast-loading asset for the landing page hero.

Run from the repo root:  backend/venv/bin/python scripts/bake_hero_model.py
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "public/models/atlas"
RAW_OUT = ATLAS / "hero-body-raw.glb"
FINAL_OUT = ATLAS / "hero-body.glb"

HERO_EXCLUDED_SYSTEMS = frozenset({"reproductive"})


def read_system_colours() -> dict[str, tuple[str, str]]:
    src = (ROOT / "src/types/atlas.ts").read_text()
    block = src[src.index("ATLAS_SYSTEMS"):]
    found = re.findall(
        r"id:\s*'([a-z]+)',\s*\n\s*name:\s*'([^']+)',\s*\n\s*color:\s*'(#[0-9a-fA-F]{6})'",
        block,
    )
    return {i: (name, hexcol) for i, name, hexcol in found}


def hex_to_rgb(h: str) -> list[float]:
    h = h.lstrip("#")
    srgb = np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return [float(v) for v in lin]


def load_and_merge_systems(meta: dict) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode parts and merge them per system with accumulated index offsets and native normals."""
    by_chunk: dict[int, list[dict]] = {}
    for p in meta["parts"]:
        if p["system"] in HERO_EXCLUDED_SYSTEMS:
            continue
        by_chunk.setdefault(p["chunk"], []).append(p)

    sys_pos: dict[str, list[np.ndarray]] = {}
    sys_nrm: dict[str, list[np.ndarray]] = {}
    sys_idx: dict[str, list[np.ndarray]] = {}
    sys_vcount: dict[str, int] = {}

    for ci, chunkspec in enumerate(meta["chunks"]):
        if ci not in by_chunk:
            continue

        raw_path = ROOT / "public" / chunkspec["gzip"].lstrip("/")
        if not raw_path.exists():
            raw_path = ATLAS / f"body-{ci}.bin"
            raw = raw_path.read_bytes()
        else:
            raw = gzip.decompress(raw_path.read_bytes())

        for p in by_chunk[ci]:
            s = p["system"]
            if s not in sys_pos:
                sys_pos[s] = []
                sys_nrm[s] = []
                sys_idx[s] = []
                sys_vcount[s] = 0

            v_count = p["vertexCount"]
            i_count = p["indexCount"]

            pos = np.frombuffer(
                raw, dtype="<f4", count=v_count * 3, offset=p["positions"]
            ).reshape(-1, 3)

            # Native author normals are stored as normalized Int16
            nrm_raw = np.frombuffer(
                raw, dtype="<i2", count=v_count * 3, offset=p["normals"]
            ).reshape(-1, 3)
            nrm = nrm_raw.astype(np.float32) / 32767.0

            # Crucial: offset indices by system accumulated vertex count
            idx = np.frombuffer(
                raw, dtype="<u4", count=i_count, offset=p["indices"]
            ).copy()
            idx += sys_vcount[s]

            sys_pos[s].append(pos)
            sys_nrm[s].append(nrm)
            sys_idx[s].append(idx)
            sys_vcount[s] += v_count

    merged = {}
    for s in sorted(sys_pos.keys()):
        merged[s] = (
            np.concatenate(sys_pos[s]).astype("<f4"),
            np.concatenate(sys_nrm[s]).astype("<f4"),
            np.concatenate(sys_idx[s]).astype("<u4"),
        )
    return merged


def write_glb(path: Path, systems: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], colours: dict) -> None:
    """Pack systems into one glTF 2.0 binary mesh with per-system primitives."""
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

    for s, (pos, nrm, tri) in systems.items():
        pv = add_view(pos.tobytes(), 34962)
        accessors.append({
            "bufferView": pv, "componentType": 5126, "count": int(pos.shape[0]),
            "type": "VEC3", "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()
        })
        a_pos = len(accessors) - 1

        nv = add_view(nrm.tobytes(), 34962)
        accessors.append({
            "bufferView": nv, "componentType": 5126, "count": int(nrm.shape[0]),
            "type": "VEC3"
        })
        a_nrm = len(accessors) - 1

        iv = add_view(tri.tobytes(), 34963)
        accessors.append({
            "bufferView": iv, "componentType": 5125, "count": int(tri.size),
            "type": "SCALAR"
        })
        a_idx = len(accessors) - 1

        name, hexcol = colours.get(s, (s.title(), "#aebbb8"))
        rgb = hex_to_rgb(hexcol)

        is_skin = (s == "integumentary")
        mat_def = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*rgb, 0.18 if is_skin else 1.0],
                "metallicFactor": 0.05,
                "roughnessFactor": 0.55,
            },
            "doubleSided": True,
        }
        if is_skin:
            mat_def["alphaMode"] = "BLEND"

        materials.append(mat_def)
        mesh_prims.append({
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm},
            "indices": a_idx,
            "material": len(materials) - 1,
        })

    gltf = {
        "asset": {"version": "2.0", "generator": "OpenMed Clean Hero Bake"},
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
    print(f"Decoded {len(colours)} system palette entries.")

    systems = load_and_merge_systems(meta)
    total_tri = sum(t.size // 3 for _, _, t in systems.values())
    total_vert = sum(p.shape[0] for p, _, _ in systems.values())
    print(f"Merged {len(systems)} systems: {total_vert:,} vertices, {total_tri:,} triangles.")

    write_glb(RAW_OUT, systems, colours)
    print(f"Wrote uncompressed model: {RAW_OUT} ({RAW_OUT.stat().st_size / 1e6:.2f} MB)")

    # Run gltfpack simplification & compression if available
    gltfpack_cmd = shutil.which("gltfpack") or "npx"
    cmd = (
        ["gltfpack", "-i", str(RAW_OUT), "-o", str(FINAL_OUT), "-si", "0.05", "-slb", "-c"]
        if gltfpack_cmd == "gltfpack"
        else ["npx", "gltfpack", "-i", str(RAW_OUT), "-o", str(FINAL_OUT), "-si", "0.05", "-slb", "-c"]
    )
    print(f"Running optimization: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and FINAL_OUT.exists():
        print(f"Successfully baked optimized hero model: {FINAL_OUT} ({FINAL_OUT.stat().st_size / 1e6:.2f} MB)")
        RAW_OUT.unlink(missing_ok=True)
    else:
        print(f"Optimization warning (keeping raw): {res.stderr}")
        shutil.move(RAW_OUT, FINAL_OUT)


if __name__ == "__main__":
    main()
