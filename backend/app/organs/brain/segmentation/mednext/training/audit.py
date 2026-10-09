"""Dataset hygiene — inspection, duplicate resolution and cleaning decisions.

A split built on a dirty dataset is worse than no split, so this runs before
any split exists. One parallel pass reads every image/mask pair once and
records everything the checks need; the checks then run on those records.

Checks (each is logged as PASS / WARN / FAIL / INFO and kept in the report):

  H01 pair integrity           every image has exactly one mask
  H02 readability              no unreadable files
  H03 geometry                 image/mask shape match; resolution census
  H04 channels                 channel census and R == G == B (grayscale-safe)
  H05 mask values              strictly binary, ambiguous-pixel census
  H06 mask extent              no empty, full-frame or sub-visible masks
  H07 mask alignment           mask foreground sits on anatomy
  H08 mask morphology          components, holes, border contact, solidity
  H09 image content            blank / low-contrast / saturated frames
  H10 exact duplicates         SHA-256 of bytes and of decoded pixels
  H11 conflicting duplicates   identical image carrying different masks
  H12 near duplicates          DCT pHash, pixel-confirmed
  H13 group integrity          filename group keys parse; group size census

The cleaning policy mirrors the earlier LiteMedSAM clean-up so both models are
trained on the same notion of "clean": byte/pixel-identical frames are collapsed
to the best-aligned member; the same image under two different lesions is
dropped entirely (nothing says which label is right); badly aligned masks and
unusable files are dropped. Nothing is dropped silently — every removal is a
row in ``dropped_files.csv`` with a reason.

Vocabulary: OpenMed group keys come from filenames and are *lesion/series*
keys, never claimed to be verified patient identifiers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Optional

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

logger = logging.getLogger("audit")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
THUMB_SIDE = 32
ROWS_VERSION = 3


# ─────────────────────────────────────────────────────────────────────────────
# Source description
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceSpec:
    name: str
    root: str
    images_dir: str = "images"
    masks_dir: str = "masks"
    mask_ext: str = ".png"
    mask_suffix: str = ""
    group_of: Optional[Callable[[str], str]] = None
    group_kind: str = "none"            # lesion | block | none
    group_is_subject_verified: bool = False

    @property
    def images_path(self) -> str:
        return os.path.join(self.root, self.images_dir)

    @property
    def masks_path(self) -> str:
        return os.path.join(self.root, self.masks_dir)


def _mask_stems(spec: SourceSpec) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in os.listdir(spec.masks_path):
        if not name.lower().endswith(spec.mask_ext):
            continue
        base = os.path.splitext(name)[0]
        if spec.mask_suffix:
            if not base.endswith(spec.mask_suffix):
                continue
            base = base[: -len(spec.mask_suffix)]
        found[base] = name
    return found


def _image_stems(spec: SourceSpec) -> dict[str, str]:
    return {
        os.path.splitext(f)[0]: f
        for f in os.listdir(spec.images_path)
        if f.lower().endswith(IMG_EXT)
    }


def list_stems(spec: SourceSpec) -> list[str]:
    return sorted(set(_image_stems(spec)) & set(_mask_stems(spec)))


def numeric_sorted(stems: list[str]) -> list[str]:
    def key(stem: str):
        return (0, int(stem), stem) if stem.isdigit() else (1, 0, stem)

    return sorted(stems, key=key)


def orphans(spec: SourceSpec) -> dict[str, list[str]]:
    images, masks = set(_image_stems(spec)), set(_mask_stems(spec))
    return {
        "images_without_mask": sorted(images - masks),
        "masks_without_image": sorted(masks - images),
    }


def resolve_image_path(spec: SourceSpec, stem: str) -> str:
    filename = _image_stems(spec).get(stem)
    if filename is None:
        raise FileNotFoundError(f"No image for stem {stem!r} under {spec.images_path}")
    return os.path.join(spec.images_path, filename)


def resolve_mask_path(spec: SourceSpec, stem: str) -> str:
    return os.path.join(spec.masks_path, stem + spec.mask_suffix + spec.mask_ext)


def paths_for(spec: SourceSpec, stems: list[str]) -> tuple[list[str], list[str]]:
    """Image and mask paths for ``stems`` using a single directory listing."""
    images = _image_stems(spec)
    return (
        [os.path.join(spec.images_path, images[s]) for s in stems],
        [resolve_mask_path(spec, s) for s in stems],
    )


def detect_layout(root: str, name: str, min_pairs: int = 50) -> dict[str, str]:
    """Work out where images and masks live inside a mounted dataset."""
    candidates = [
        ("images", "masks", ""),
        ("images", "masks", "_mask"),
        ("png_dataset", "png_dataset", "_mask"),
        (".", ".", "_mask"),
        ("images", "labels", ""),
    ]

    def usable(images_dir: str, masks_dir: str, suffix: str) -> int:
        if not (
            os.path.isdir(os.path.join(root, images_dir))
            and os.path.isdir(os.path.join(root, masks_dir))
        ):
            return 0
        spec = SourceSpec(name, root, images_dir, masks_dir, mask_suffix=suffix)
        try:
            return len(list_stems(spec))
        except OSError:
            return 0

    for images_dir, masks_dir, suffix in candidates:
        count = usable(images_dir, masks_dir, suffix)
        if count >= min_pairs:
            logger.info(
                "[%s] layout: images=%s masks=%s mask_suffix=%r (%d paired stems)",
                name, images_dir, masks_dir, suffix, count,
            )
            return {"images_dir": images_dir, "masks_dir": masks_dir, "mask_suffix": suffix}

    best: Optional[tuple[int, dict[str, str]]] = None
    for entry in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, entry)):
            continue
        for suffix in ("_mask", ""):
            count = usable(entry, entry, suffix)
            if count >= min_pairs and (best is None or count > best[0]):
                best = (count, {"images_dir": entry, "masks_dir": entry, "mask_suffix": suffix})
    if best is not None:
        return best[1]

    raise RuntimeError(
        f"Could not determine the layout of {name} at {root}. Top level: "
        f"{sorted(os.listdir(root))[:20]}. Expected images/ + masks/ (or png_dataset/ "
        "with <stem>_mask.png)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair inspection (module level so multiprocessing can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def phash64(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash of a grayscale image."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def _popcount(values: np.ndarray) -> np.ndarray:
    u = values.astype(np.uint64)
    u = u - ((u >> np.uint64(1)) & np.uint64(0x5555555555555555))
    u = (u & np.uint64(0x3333333333333333)) + ((u >> np.uint64(2)) & np.uint64(0x3333333333333333))
    u = (u + (u >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((u * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def _head_mask(gray: np.ndarray) -> np.ndarray:
    head = (gray > max(10, 0.10 * float(gray.max()))).astype(np.uint8)
    if not head.any():
        return head
    n, labels, stats, _ = cv2.connectedComponentsWithStats(head, connectivity=8)
    if n > 1:
        head = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return binary_fill_holes(head).astype(np.uint8)


def inspect_pair(args: tuple[str, str, str, float]) -> dict[str, Any]:
    """Everything the hygiene checks need about one image/mask pair."""
    cv2.setNumThreads(1)
    stem, image_path, mask_path, align_min = args
    row: dict[str, Any] = {"stem": stem, "ok": False}

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        row["unreadable"] = True
        return row

    row["sha_image"] = _sha256_file(image_path)
    row["sha_mask"] = _sha256_file(mask_path)

    n_channels = image.shape[2] if image.ndim == 3 else 1
    if n_channels > 1:
        colour = image[:, :, :3].astype(np.int16)
        diff = np.maximum(
            abs(colour[:, :, 0] - colour[:, :, 1]), abs(colour[:, :, 1] - colour[:, :, 2])
        )
        row["channel_spread"] = int(diff.max())
        row["colour_frac"] = float((diff > 10).mean())
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        row["channel_spread"] = 0
        row["colour_frac"] = 0.0
        gray = image
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    values, counts = np.unique(mask, return_counts=True)
    binary = (mask >= 128).astype(np.uint8)
    fg = int(binary.sum())
    total = int(binary.size)
    clean_values = np.isin(values, (0, 255))

    row.update({
        "unreadable": False,
        "image_hw": [int(gray.shape[0]), int(gray.shape[1])],
        "mask_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "shape_match": gray.shape[:2] == mask.shape[:2],
        "n_channels": int(n_channels),
        "image_dtype": str(image.dtype),
        "mask_dtype": str(mask.dtype),
        "mask_values": [int(v) for v in values[:6]],
        "is_binary": bool(clean_values.all()),
        "ambiguous_frac": float(counts[~clean_values].sum() / total),
        "fg_px": fg,
        "fg_frac": fg / total,
        "empty": fg == 0,
        "full": fg == total,
        "touches_border": bool(
            binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any()
        ),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p99": float(np.percentile(gray, 99)),
        "gray_max": int(gray.max()),
        "zero_frac": float((gray == 0).mean()),
        "sat_frac": float((gray >= 250).mean()),
        "sha_pixels": hashlib.sha256(gray.tobytes() + str(gray.shape).encode()).hexdigest(),
        "sha_maskpix": hashlib.sha256(binary.tobytes() + str(binary.shape).encode()).hexdigest(),
        "phash": phash64(gray),
    })

    if fg:
        ys, xs = np.nonzero(binary)
        row["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
        row["n_components"] = int(n_labels - 1)
        row["largest_frac"] = areas[0] / fg
        row["n_tiny"] = int(sum(1 for a in areas if a < 16))

        inverse = (1 - binary).astype(np.uint8)
        n_inv, _, inv_stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
        holes = 0
        for idx in range(1, n_inv):
            x, y, w, h, area = inv_stats[idx]
            if x == 0 or y == 0 or x + w >= binary.shape[1] or y + h >= binary.shape[0]:
                continue
            holes += int(area >= 16)
        row["n_holes"] = holes

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        solidity = circularity = 0.0
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            perimeter = cv2.arcLength(biggest, True)
            hull_area = cv2.contourArea(cv2.convexHull(biggest))
            solidity = float(area / hull_area) if hull_area > 0 else 0.0
            circularity = float(4 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0
        row["solidity"] = solidity
        row["circularity"] = circularity

        head = _head_mask(gray)
        row["head_frac"] = float(head.mean())
        row["inside_frac"] = float((binary & head).sum() / fg) if head.shape == binary.shape else 0.0
    else:
        row.update({
            "n_components": 0, "largest_frac": 0.0, "n_tiny": 0, "n_holes": 0,
            "solidity": 0.0, "circularity": 0.0, "inside_frac": 0.0, "head_frac": 0.0,
        })

    row["align_ok"] = bool(row["inside_frac"] >= align_min)
    row["blank"] = bool(row["gray_std"] < 1e-6)

    thumb = cv2.resize(gray, (THUMB_SIDE, THUMB_SIDE), interpolation=cv2.INTER_AREA)
    row["_thumb"] = thumb
    row["ok"] = True
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Report plumbing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Check:
    code: str
    name: str
    status: str            # PASS | WARN | FAIL | INFO
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "name": self.name, "status": self.status,
                "detail": self.detail, "data": self.data}


class CheckLog:
    """Collects checks and logs each one the moment it is made."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.checks: list[Check] = []

    def add(self, code: str, name: str, status: str, detail: str, **data: Any) -> Check:
        check = Check(code, name, status, detail, data)
        self.checks.append(check)
        level = {"PASS": logging.INFO, "INFO": logging.INFO,
                 "WARN": logging.WARNING, "FAIL": logging.ERROR}[status]
        logger.log(level, "[%s] %s %-26s %-4s %s", self.source, code, name, status, detail)
        return check

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.checks]


@dataclass
class SourceAudit:
    spec: SourceSpec
    stems: list[str]
    rows: dict[str, dict[str, Any]]
    thumbs: dict[str, np.ndarray]
    checks: CheckLog
    drops: dict[str, dict[str, str]]
    kept: list[str]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.spec.name,
            "n_paired": len(self.stems),
            "n_kept": len(self.kept),
            "n_dropped": len(self.drops),
            "checks": self.checks.to_list(),
            "summary": self.summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inspection with caching
# ─────────────────────────────────────────────────────────────────────────────

def _stem_digest(stems: list[str]) -> str:
    return hashlib.sha256("\n".join(stems).encode()).hexdigest()[:16]


def inspect_source(
    spec: SourceSpec,
    stems: list[str],
    *,
    workers: int,
    align_min: float,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    digest = _stem_digest(stems)
    rows_path = thumbs_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        rows_path = os.path.join(cache_dir, f"{spec.name}_rows.json")
        thumbs_path = os.path.join(cache_dir, f"{spec.name}_thumbs.npz")
        if not force and os.path.isfile(rows_path) and os.path.isfile(thumbs_path):
            with open(rows_path) as handle:
                cached = json.load(handle)
            if cached.get("digest") == digest and cached.get("version") == ROWS_VERSION:
                with np.load(thumbs_path) as npz:
                    thumbs = {s: npz[f"t{i}"] for i, s in enumerate(cached["order"])}
                logger.info("[%s] reusing cached inspection (%d pairs)", spec.name, len(stems))
                return cached["rows"], thumbs

    image_paths, mask_paths = paths_for(spec, stems)
    jobs = [
        (s, ip, mp, align_min)
        for s, ip, mp in zip(stems, image_paths, mask_paths, strict=True)
    ]
    logger.info("[%s] inspecting %d pairs with %d workers", spec.name, len(jobs), workers)
    results: list[dict[str, Any]] = []
    step = max(1, len(jobs) // 10)
    if workers > 1:
        with Pool(workers) as pool:
            for done, result in enumerate(pool.imap(inspect_pair, jobs, chunksize=16), 1):
                results.append(result)
                if done % step == 0:
                    logger.info("[%s] inspected %d / %d", spec.name, done, len(jobs))
    else:
        for done, job in enumerate(jobs, 1):
            results.append(inspect_pair(job))
            if done % step == 0:
                logger.info("[%s] inspected %d / %d", spec.name, done, len(jobs))

    rows: dict[str, dict[str, Any]] = {}
    thumbs: dict[str, np.ndarray] = {}
    for result in results:
        thumb = result.pop("_thumb", None)
        rows[result["stem"]] = result
        if thumb is not None:
            thumbs[result["stem"]] = thumb

    if rows_path and thumbs_path:
        order = [s for s in stems if s in thumbs]
        with open(rows_path, "w") as handle:
            json.dump({"digest": digest, "version": ROWS_VERSION, "order": order, "rows": rows}, handle)
        np.savez_compressed(thumbs_path, **{f"t{i}": thumbs[s] for i, s in enumerate(order)})
    return rows, thumbs


# ─────────────────────────────────────────────────────────────────────────────
# Near-duplicate machinery
# ─────────────────────────────────────────────────────────────────────────────

def hamming_pairs(
    hashes: np.ndarray, max_distance: int, chunk: int = 512
) -> list[tuple[int, int, int]]:
    """All index pairs ``i < j`` whose hashes differ in at most ``max_distance`` bits."""
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        dist = _popcount(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows, cols, strict=True):
            i, j = start + int(r), int(c)
            if i < j:
                pairs.append((i, j, int(dist[r, c])))
    return pairs


def pixel_mae(path_a: str, path_b: str) -> Optional[float]:
    a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def phash_census(
    stems: list[str],
    rows: dict[str, dict[str, Any]],
    group_of: Optional[Callable[[str], str]],
    max_distance: int,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]], np.ndarray, list[tuple[int, int, int]]]:
    """pHash pair counts per threshold, split into same-group and cross-group."""
    usable = [s for s in stems if rows[s].get("ok")]
    hashes = np.array([rows[s]["phash"] for s in usable], dtype=np.uint64)
    pairs = hamming_pairs(hashes, max_distance)

    counts = {f"hamming<={t}": {"within_group": 0, "cross_group": 0} for t in range(max_distance + 1)}
    examples: list[dict[str, Any]] = []
    for i, j, d in pairs:
        a, b = usable[i], usable[j]
        if group_of is not None:
            same = group_of(a) == group_of(b)
        else:
            try:
                same = abs(int(a) - int(b)) <= 4
            except ValueError:
                same = False
        for t in range(d, max_distance + 1):
            counts[f"hamming<={t}"]["within_group" if same else "cross_group"] += 1
        if not same and len(examples) < 20:
            examples.append({"a": a, "b": b, "hamming": d})
    return counts, examples, hashes, pairs


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning decisions
# ─────────────────────────────────────────────────────────────────────────────

def _rank(row: dict[str, Any], stem: str) -> tuple:
    """Best-member ordering: better aligned, then more foreground, then name."""
    return (-round(row.get("inside_frac", 0.0), 6), -row.get("fg_px", 0), stem)


def decide_drops(
    stems: list[str],
    rows: dict[str, dict[str, Any]],
    group_of: Optional[Callable[[str], str]],
    twin_pairs: list[tuple[str, str]],
    *,
    align_min: float,
    min_image_std: float,
) -> dict[str, dict[str, str]]:
    """Apply the cleaning policy. Returns ``stem -> {reason, kept_as}``."""
    drops: dict[str, dict[str, str]] = {}

    def drop(stem: str, reason: str, kept_as: str = "") -> None:
        drops.setdefault(stem, {"reason": reason, "kept_as": kept_as})

    def lesion(stem: str) -> str:
        return str(group_of(stem)) if group_of else "-"

    # A. files that cannot be used at all
    for stem in stems:
        row = rows[stem]
        if not row.get("ok"):
            drop(stem, "unreadable file")
        elif not row["shape_match"]:
            drop(stem, "image and mask shapes differ")
        elif row["empty"]:
            drop(stem, "empty mask")
        elif row["full"]:
            drop(stem, "full-frame mask")
        elif row["gray_std"] < min_image_std:
            drop(stem, f"blank or low-contrast image (std < {min_image_std:g})")

    # B1. identical bytes / identical decoded pixels (an equivalence relation)
    alive = [s for s in stems if s not in drops]
    index = {s: i for i, s in enumerate(alive)}
    union = UnionFind(len(alive))
    linked_by: dict[str, str] = {}
    for key, label in (("sha_image", "identical bytes"), ("sha_pixels", "identical decoded pixels")):
        buckets: dict[str, list[str]] = defaultdict(list)
        for stem in alive:
            digest = rows[stem].get(key)
            if digest:
                buckets[digest].append(stem)
        for members in buckets.values():
            for other in members[1:]:
                union.union(index[members[0]], index[other])
                linked_by.setdefault(other, label)

    clusters: dict[int, list[str]] = defaultdict(list)
    for stem in alive:
        clusters[union.find(index[stem])].append(stem)
    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda s: _rank(rows[s], s))
        how = linked_by.get(members[1], "identical image")
        lesions = {lesion(s) for s in members}
        if len(lesions) > 1:
            for s in members:
                drop(s, f"{how} claiming {len(lesions)} different lesions (label unknowable)")
            continue
        masks = {rows[s]["sha_maskpix"] for s in members}
        reason = (
            f"duplicate image, same mask ({how})"
            if len(masks) == 1
            else f"duplicate image, conflicting masks within one lesion ({how})"
        )
        for s in members[1:]:
            drop(s, reason, kept_as=members[0])

    # B2. pixel-confirmed near-identical pairs: greedy selection, never transitive,
    # so a slowly changing series cannot collapse into one survivor.
    survivors = [s for s in alive if s not in drops]
    alive_set = set(survivors)
    neighbours: dict[str, set[str]] = defaultdict(set)
    for a, b in twin_pairs:
        if a in alive_set and b in alive_set:
            neighbours[a].add(b)
            neighbours[b].add(a)
    near = "near-identical pixels (pHash, pixel-confirmed)"
    for stem in sorted(survivors, key=lambda s: _rank(rows[s], s)):
        if stem in drops:
            continue
        for other in sorted(neighbours.get(stem, ()), key=lambda s: _rank(rows[s], s)):
            if other in drops:
                continue
            if lesion(stem) != lesion(other):
                drop(stem, f"{near} claiming 2 different lesions (label unknowable)")
                drop(other, f"{near} claiming 2 different lesions (label unknowable)")
                break
            if rows[stem]["sha_maskpix"] == rows[other]["sha_maskpix"]:
                continue
            drop(other, f"near-duplicate image, conflicting mask ({near})", kept_as=stem)

    # C. alignment, judged on the member that survived de-duplication
    for stem in stems:
        if stem not in drops and rows[stem]["inside_frac"] < align_min:
            drop(stem, f"mask sits below {align_min:.0%} on anatomy")
    return drops


def write_drops(drops: dict[str, dict[str, str]], source: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "stem", "reason", "kept_as"])
        for stem in sorted(drops):
            writer.writerow([source, stem, drops[stem]["reason"], drops[stem]["kept_as"]])


# ─────────────────────────────────────────────────────────────────────────────
# The audit
# ─────────────────────────────────────────────────────────────────────────────

def _dist(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "p5": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()), "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def audit_source(
    spec: SourceSpec,
    cfg: Any,
    *,
    workers: int,
    out_dir: str,
    force: bool = False,
) -> SourceAudit:
    """Run hygiene checks H01-H13 on one source and decide what to drop."""
    log = CheckLog(spec.name)
    stems = list_stems(spec)
    if getattr(cfg, "limit_per_source", 0):
        stems = numeric_sorted(stems)[: cfg.limit_per_source]
        logger.warning("[%s] --limit: auditing only the first %d stems", spec.name, len(stems))
    orphan_map = orphans(spec)
    n_orphans = len(orphan_map["images_without_mask"]) + len(orphan_map["masks_without_image"])

    # H01
    log.add(
        "H01", "pair integrity", "PASS" if n_orphans == 0 else "WARN",
        f"{len(stems)} paired stems; {len(orphan_map['images_without_mask'])} images without mask, "
        f"{len(orphan_map['masks_without_image'])} masks without image",
        orphans={k: v[:25] for k, v in orphan_map.items()}, n_orphans=n_orphans,
    )
    if len(stems) < 50:
        raise RuntimeError(f"[{spec.name}] only {len(stems)} paired stems — refusing to continue.")

    rows, thumbs = inspect_source(
        spec, stems, workers=workers, align_min=cfg.align_min_inside,
        cache_dir=os.path.join(out_dir, "audit"), force=force,
    )
    readable = [s for s in stems if rows[s].get("ok")]
    r = [rows[s] for s in readable]

    # H02
    unreadable = [s for s in stems if not rows[s].get("ok")]
    log.add("H02", "readability", "PASS" if not unreadable else "FAIL",
            f"{len(unreadable)} unreadable of {len(stems)}", examples=unreadable[:20])

    # H03
    mismatched = [x["stem"] for x in r if not x["shape_match"]]
    resolutions = Counter(tuple(x["image_hw"]) for x in r)
    log.add("H03", "geometry", "PASS" if not mismatched else "FAIL",
            f"{len(mismatched)} image/mask shape mismatches; resolutions "
            f"{ {f'{h}x{w}': n for (h, w), n in resolutions.most_common(4)} }",
            mismatched=mismatched[:20], resolutions={f"{h}x{w}": n for (h, w), n in resolutions.items()})

    # H04
    channels = Counter(x["n_channels"] for x in r)
    marks = [x["stem"] for x in r if x["colour_frac"] > 0.0]
    colour = [x["stem"] for x in r if x["colour_frac"] > 0.02]
    log.add("H04", "channels", "PASS" if not colour else "WARN",
            f"channel census {dict(channels)}; {len(marks)} frames hold a few coloured pixels "
            f"(annotation marks, <2% of the frame, harmless to a 1-channel model); "
            f"{len(colour)} frames are genuinely coloured (>2%)",
            channels=dict(channels), annotation_frames=len(marks), colour_examples=colour[:20])

    # H05
    non_binary = [x["stem"] for x in r if not x["is_binary"]]
    amb = [x["ambiguous_frac"] for x in r]
    log.add("H05", "mask values", "PASS" if not non_binary else "WARN",
            f"{len(non_binary)} masks hold values other than {{0,255}} "
            f"(max ambiguous fraction {max(amb) if amb else 0:.4f}); masks binarised at >=128",
            examples=non_binary[:20])

    # H06
    empty = [x["stem"] for x in r if x["empty"]]
    full = [x["stem"] for x in r if x["full"]]
    tiny = [x["stem"] for x in r if 0 < x["fg_frac"] < cfg.tiny_mask_frac]
    fg_dist = _dist([x["fg_frac"] for x in r])
    log.add("H06", "mask extent", "PASS" if not (empty or full) else "WARN",
            f"{len(empty)} empty, {len(full)} full-frame, {len(tiny)} below {cfg.tiny_mask_frac:.2%} of frame; "
            f"fg fraction min {fg_dist['min']:.4f} median {fg_dist['median']:.4f} max {fg_dist['max']:.4f}",
            fg_frac=fg_dist, tiny_examples=tiny[:20])

    # H07
    misaligned = [x["stem"] for x in r if not x["empty"] and not x["align_ok"]]
    inside = _dist([x["inside_frac"] for x in r if not x["empty"]])
    log.add("H07", "mask alignment", "PASS" if not misaligned else "WARN",
            f"{len(misaligned)} masks below {cfg.align_min_inside:.0%} on anatomy "
            f"(min {inside['min']:.3f}, median {inside['median']:.3f})",
            inside=inside, examples=misaligned[:20])

    # H08
    comps = [x["n_components"] for x in r if not x["empty"]]
    holes = [x["n_holes"] for x in r if not x["empty"]]
    border = sum(1 for x in r if x["touches_border"])
    log.add("H08", "mask morphology", "INFO",
            f"components median {np.median(comps) if comps else 0:.0f} max {max(comps) if comps else 0}; "
            f"{sum(1 for c in comps if c > 2)} fragmented (>2); {sum(1 for h in holes if h)} with holes; "
            f"{border} touch the border; solidity median "
            f"{np.median([x['solidity'] for x in r if not x['empty']] or [0]):.3f}",
            components=_dist([float(c) for c in comps]),
            solidity=_dist([x["solidity"] for x in r if not x["empty"]]),
            circularity=_dist([x["circularity"] for x in r if not x["empty"]]),
            border_touching=border)

    # H09
    low = [x["stem"] for x in r if x["gray_std"] < cfg.min_image_std]
    blank = [x["stem"] for x in r if x["blank"]]
    saturated = [x["stem"] for x in r if x["sat_frac"] > 0.25]
    log.add("H09", "image content", "PASS" if not (low or blank) else "WARN",
            f"{len(blank)} constant frames, {len(low)} with std < {cfg.min_image_std:g}, "
            f"{len(saturated)} with >25% saturated pixels; gray mean median "
            f"{np.median([x['gray_mean'] for x in r]):.1f}, std median "
            f"{np.median([x['gray_std'] for x in r]):.1f}",
            gray_mean=_dist([x["gray_mean"] for x in r]),
            gray_std=_dist([x["gray_std"] for x in r]),
            zero_frac=_dist([x["zero_frac"] for x in r]),
            low_contrast_examples=low[:20], saturated_examples=saturated[:20])

    # H10 / H11
    def clusters_of(key: str) -> list[list[str]]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for x in r:
            if x.get(key):
                buckets[x[key]].append(x["stem"])
        return [m for m in buckets.values() if len(m) > 1]

    byte_clusters = clusters_of("sha_image")
    pixel_clusters = clusters_of("sha_pixels")
    log.add("H10", "exact duplicates",
            "PASS" if not (byte_clusters or pixel_clusters) else "WARN",
            f"{len(byte_clusters)} byte-identical image groups ({sum(len(c) for c in byte_clusters)} files); "
            f"{len(pixel_clusters)} decoded-pixel-identical groups ({sum(len(c) for c in pixel_clusters)} files)",
            byte_groups=len(byte_clusters), pixel_groups=len(pixel_clusters),
            examples=[c[:4] for c in (byte_clusters or pixel_clusters)[:10]])

    group_of = spec.group_of
    conflicting = [
        c for c in pixel_clusters if len({rows[s]["sha_maskpix"] for s in c}) > 1
    ]
    cross_lesion = [c for c in pixel_clusters if group_of and len({group_of(s) for s in c}) > 1]
    log.add("H11", "conflicting duplicates", "PASS" if not conflicting else "WARN",
            f"{len(conflicting)} identical images carry different masks "
            f"({sum(len(c) for c in conflicting)} files); {len(cross_lesion)} span more than one lesion",
            conflicting=len(conflicting), cross_lesion=len(cross_lesion),
            examples=[c[:4] for c in conflicting[:10]])

    # H12
    counts, examples, hashes, pairs = phash_census(
        readable, rows, group_of, cfg.phash_hamming_max
    )
    candidates = [(i, j) for i, j, d in pairs if d <= cfg.phash_dup_hamming]
    cap = 20000
    if len(candidates) > cap:
        logger.warning("[%s] %d pHash candidates; confirming the first %d", spec.name, len(candidates), cap)
        candidates = candidates[:cap]
    image_paths, _ = paths_for(spec, readable)
    twin_pairs: list[tuple[str, str]] = []
    adjacent_distinct = 0
    for i, j in candidates:
        mae = pixel_mae(image_paths[i], image_paths[j])
        if mae is not None and mae < cfg.dup_pixel_mae:
            twin_pairs.append((readable[i], readable[j]))
        else:
            adjacent_distinct += 1
    cross = counts[f"hamming<={cfg.phash_hamming_max}"]["cross_group"]
    log.add("H12", "near duplicates", "PASS" if not twin_pairs else "WARN",
            f"pHash pairs <= {cfg.phash_hamming_max}: {counts[f'hamming<={cfg.phash_hamming_max}']}; "
            f"{len(twin_pairs)} pixel-confirmed twins (MAE < {cfg.dup_pixel_mae:g}), "
            f"{adjacent_distinct} similar-but-distinct (neighbouring slices, kept)",
            counts=counts, cross_group_examples=examples,
            confirmed_twins=len(twin_pairs), similar_distinct=adjacent_distinct)

    # H13
    group_summary: dict[str, Any] = {"kind": spec.group_kind,
                                     "subject_verified": spec.group_is_subject_verified}
    if group_of is not None:
        grouped: dict[str, list[str]] = defaultdict(list)
        for s in stems:
            grouped[str(group_of(s))].append(s)
        sizes = np.array([len(v) for v in grouped.values()])
        group_summary.update({
            "n_groups": len(grouped),
            "slices_per_group": {"median": float(np.median(sizes)), "mean": float(sizes.mean()),
                                 "max": int(sizes.max())},
            "singletons": int((sizes == 1).sum()),
        })
        log.add("H13", "group integrity", "INFO",
                f"{len(grouped)} {spec.group_kind} groups; slices/group median {np.median(sizes):.0f} "
                f"max {sizes.max()}; {int((sizes == 1).sum())} singletons. Filename-derived, "
                "NOT a verified patient id (merging distinct patients is safe, splitting one is not)",
                **group_summary)
    else:
        log.add("H13", "group integrity", "WARN",
                "no recoverable group key — split will use snapped contiguous blocks", **group_summary)

    drops = decide_drops(
        stems, rows, group_of, twin_pairs,
        align_min=cfg.align_min_inside, min_image_std=cfg.min_image_std,
    )
    reasons = Counter(
        d["reason"].split(" (")[0] for d in drops.values()
    )
    kept = [s for s in stems if s not in drops]
    log.add("CLEAN", "cleaning policy", "INFO" if not drops else "WARN",
            f"kept {len(kept)} of {len(stems)}; dropped {len(drops)}: {dict(reasons)}",
            reasons=dict(reasons), n_kept=len(kept), n_dropped=len(drops))

    verify_clean(kept, rows, group_of, log)

    write_drops(drops, spec.name, os.path.join(out_dir, "logs", f"dropped_{spec.name}.csv"))

    summary = {
        "n_paired": len(stems), "n_kept": len(kept), "n_dropped": len(drops),
        "drop_reasons": dict(reasons), "groups": group_summary,
        "fg_frac": fg_dist, "gray_mean": _dist([x["gray_mean"] for x in r]),
        "gray_std": _dist([x["gray_std"] for x in r]),
        "phash": counts, "twin_pairs": len(twin_pairs),
    }
    return SourceAudit(spec, stems, rows, thumbs, log, drops, kept, summary)


def verify_clean(
    kept: list[str],
    rows: dict[str, dict[str, Any]],
    group_of: Optional[Callable[[str], str]],
    log: CheckLog,
) -> None:
    """Post-clean assertions: re-run the checks that must now hold exactly."""
    kr = [rows[s] for s in kept]
    problems: list[str] = []
    if any(not x.get("ok") for x in kr):
        problems.append("unreadable files remain")
    if any(x["empty"] or x["full"] for x in kr):
        problems.append("empty/full masks remain")
    for key in ("sha_image", "sha_pixels"):
        seen: dict[str, int] = Counter(x[key] for x in kr if x.get(key))
        if any(v > 1 for v in seen.values()):
            problems.append(f"{key} duplicates remain")
    log.add("VERIFY", "post-clean state", "PASS" if not problems else "FAIL",
            "no unreadable/empty/full masks and no exact duplicates remain"
            if not problems else "; ".join(problems))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-source overlap
# ─────────────────────────────────────────────────────────────────────────────

def cross_source_overlap(
    a: SourceAudit, b: SourceAudit, cfg: Any
) -> dict[str, Any]:
    """Do two sources share any image? Perceptual hit, then pixel-level proof."""
    stems_a, stems_b = a.kept, b.kept
    sha_a = {a.rows[s]["sha_image"] for s in stems_a}
    sha_b = {b.rows[s]["sha_image"] for s in stems_b}
    pix_a = {a.rows[s]["sha_pixels"] for s in stems_a}
    pix_b = {b.rows[s]["sha_pixels"] for s in stems_b}

    va = np.array([a.rows[s]["phash"] for s in stems_a], dtype=np.uint64)
    vb = np.array([b.rows[s]["phash"] for s in stems_b], dtype=np.uint64)
    candidates: list[tuple[int, int, int]] = []
    chunk = 512
    for start in range(0, len(va), chunk):
        dist = _popcount(va[start:start + chunk, None] ^ vb[None, :])
        rr, cc = np.nonzero(dist <= cfg.phash_hamming_max)
        for r_, c_ in zip(rr, cc, strict=True):
            candidates.append((start + int(r_), int(c_), int(dist[r_, c_])))

    ia, _ = paths_for(a.spec, stems_a)
    ib, _ = paths_for(b.spec, stems_b)
    confirmed: list[dict[str, Any]] = []
    for i, j, d in candidates[:500]:
        mae = pixel_mae(ia[i], ib[j])
        confirmed.append({
            "a": stems_a[i], "b": stems_b[j], "hamming": d, "pixel_mae": mae,
            "same_image": bool(mae is not None and mae < cfg.dup_pixel_mae),
        })
    shared = [c for c in confirmed if c["same_image"]]
    result = {
        "exact_bytes_shared": len(sha_a & sha_b),
        "exact_pixels_shared": len(pix_a & pix_b),
        "phash_candidates": len(candidates),
        "confirmed_shared_images": len(shared),
        "shared_pairs": shared[:50],
        "candidates_checked": len(confirmed),
        "verdict": "DISJOINT" if not shared and not (pix_a & pix_b) else "OVERLAP",
    }
    status = "PASS" if result["verdict"] == "DISJOINT" else "FAIL"
    logger.log(
        logging.INFO if status == "PASS" else logging.ERROR,
        "[cross-source] H14 overlap %-4s %d pHash candidates, %d pixel-confirmed shared, "
        "%d exact-pixel shared -> %s",
        status, len(candidates), len(shared), len(pix_a & pix_b), result["verdict"],
    )
    return result


def drop_cross_source_overlap(
    a: SourceAudit, b: SourceAudit, overlap: dict[str, Any]
) -> int:
    """Remove the BTSC copy of any image both sources contain. Returns the count."""
    removed = 0
    for pair in overlap["shared_pairs"]:
        stem = pair["b"]
        if stem in b.kept:
            b.kept.remove(stem)
            b.drops[stem] = {"reason": f"image also present in {a.spec.name} ({pair['a']})",
                             "kept_as": pair["a"]}
            removed += 1
    return removed


def sample_stems(stems: list[str], k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return sorted(rng.sample(stems, min(k, len(stems))))
