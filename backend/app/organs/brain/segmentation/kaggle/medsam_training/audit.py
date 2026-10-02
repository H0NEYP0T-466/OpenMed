"""Stage 2 — dataset hygiene audit.

Runs before any split is built, because a split built on a dirty dataset is
worse than no split at all. Checks, in order of how badly they can ruin a
result:

  1. Pair integrity      — every image has exactly one mask, no orphans
  2. Geometry            — shape match, binary masks, no empty/full masks
  3. Exact duplicates    — SHA256 over image bytes and mask bytes
  4. Near duplicates     — DCT pHash, Hamming distance <= threshold
  5. Alignment           — mask foreground actually sits on anatomy
  6. Morphology          — component count, holes, border contact
  7. Cross-source overlap— do the two datasets share any image at all?

Every finding is returned as data *and* written to the run log. Nothing is
silently dropped: the caller decides, and the removals are recorded with a
reason per file.

A note on vocabulary, because it matters for the report: OpenMed's group key
is derived from the dataset's own filenames (sequence prefix and slice number
stripped). That makes it a **lesion** key. Whether a lesion equals a patient
is not knowable from the files, so this module never claims "patient" — it
reports ``group_is_subject_verified: False`` and the split builder carries the
same caveat forward.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Iterable, Optional

import cv2
import numpy as np

logger = logging.getLogger("audit")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ─────────────────────────────────────────────────────────────────────────────
# Source description
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceSpec:
    """One dataset on disk.

    ``group_of`` maps a stem to its grouping key. It must be a module-level
    callable (picklable) or ``None``, in which case the source has no
    recoverable grouping and the split builder falls back to contiguous blocks.
    """

    name: str
    root: str
    images_dir: str = "images"
    masks_dir: str = "masks"
    image_ext: Optional[str] = None   # None = accept any known extension
    mask_ext: str = ".png"
    # Set when masks are named ``<stem>_mask.png`` rather than ``<stem>.png``.
    # This is the layout the upstream figshare export uses, and some Kaggle
    # mirrors keep it verbatim.
    mask_suffix: str = ""
    group_of: Optional[Any] = None
    group_kind: str = "none"          # "lesion" | "block" | "none"
    group_is_subject_verified: bool = False

    @property
    def images_path(self) -> str:
        return os.path.join(self.root, self.images_dir)

    @property
    def masks_path(self) -> str:
        return os.path.join(self.root, self.masks_dir)


def _mask_stems(spec: SourceSpec) -> dict[str, str]:
    """``{stem: filename}`` for every mask, honouring ``mask_suffix``."""
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


def list_stems(spec: SourceSpec) -> list[str]:
    """Every stem that has *both* an image and a mask, sorted."""
    images = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(spec.images_path)
        if f.lower().endswith(IMG_EXT)
    }
    masks = _mask_stems(spec)
    return sorted(set(images) & set(masks))


def orphans(spec: SourceSpec) -> dict[str, list[str]]:
    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(spec.images_path)
        if f.lower().endswith(IMG_EXT)
    }
    masks = set(_mask_stems(spec))
    return {
        "images_without_mask": sorted(images - masks),
        "masks_without_image": sorted(masks - images),
    }


def resolve_image_path(spec: SourceSpec, stem: str) -> str:
    for ext in IMG_EXT:
        candidate = os.path.join(spec.images_path, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No image for stem {stem!r} under {spec.images_path}")


def resolve_mask_path(spec: SourceSpec, stem: str) -> str:
    return os.path.join(spec.masks_path, stem + spec.mask_suffix + spec.mask_ext)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel helpers (module level so multiprocessing can pickle them)
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _phash(args: tuple[str, int, int]) -> Optional[int]:
    """64-bit DCT perceptual hash. Returns None if the file is unreadable."""
    path, hash_size, factor = args
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    side = hash_size * factor
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:hash_size, :hash_size].flatten()
    bits = dct > np.median(dct[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def _popcount(values: np.ndarray) -> np.ndarray:
    """Vectorised popcount for uint64 arrays."""
    u = values.astype(np.uint64)
    u = u - ((u >> np.uint64(1)) & np.uint64(0x5555555555555555))
    u = (u & np.uint64(0x3333333333333333)) + ((u >> np.uint64(2)) & np.uint64(0x3333333333333333))
    u = (u + (u >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((u * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def _geom(args: tuple[str, str, float]) -> Optional[dict[str, Any]]:
    """Per-pair geometry, morphology and alignment."""
    image_path, mask_path, align_min = args
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None or msk is None:
        return {"unreadable": True}

    if img.ndim == 3 and img.shape[2] > 1:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        n_channels = img.shape[2]
    else:
        gray = img
        n_channels = 1

    if msk.ndim == 3:
        msk = msk[:, :, 0]

    binary = (msk > 0).astype(np.uint8)
    fg = int(binary.sum())
    total = binary.size

    result: dict[str, Any] = {
        "unreadable": False,
        "image_hw": list(img.shape[:2]),
        "mask_hw": list(msk.shape[:2]),
        "shape_match": img.shape[:2] == msk.shape[:2],
        "channels": n_channels,
        "mask_dtype": str(msk.dtype),
        "unique_mask_values": [int(v) for v in np.unique(msk)[:8]],
        "is_binary": bool(np.isin(np.unique(msk), [0, 255]).all()),
        "fg_pixels": fg,
        "fg_frac": fg / total,
        "empty": fg == 0,
        "full": fg == total,
        "touches_border": bool(
            binary[0, :].any() or binary[-1, :].any()
            or binary[:, 0].any() or binary[:, -1].any()
        ),
    }

    if fg == 0:
        result.update({"n_components": 0, "largest_frac": 0.0, "n_holes": 0,
                       "inside_frac": 0.0, "solidity": 0.0, "circularity": 0.0})
        return result

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True) if n_labels > 1 else []
    result["n_components"] = max(0, n_labels - 1)
    result["largest_frac"] = (areas[0] / fg) if areas else 0.0
    result["n_tiny"] = int(sum(1 for a in areas if a < 16))

    # Holes: background components fully enclosed by foreground.
    inv = (1 - binary).astype(np.uint8)
    n_inv, inv_labels, inv_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = 0
    for idx in range(1, n_inv):
        x, y, w, h, area = inv_stats[idx]
        if x == 0 or y == 0 or x + w >= binary.shape[1] or y + h >= binary.shape[0]:
            continue  # touches the border -> background, not a hole
        if area >= 16:
            holes += 1
    result["n_holes"] = holes

    # Shape regularity on the largest contour.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        perim = cv2.arcLength(biggest, True)
        hull_area = cv2.contourArea(cv2.convexHull(biggest))
        result["solidity"] = float(area / hull_area) if hull_area > 0 else 0.0
        result["circularity"] = float(4 * np.pi * area / (perim**2)) if perim > 0 else 0.0
    else:
        result["solidity"] = 0.0
        result["circularity"] = 0.0

    # Alignment: what fraction of the mask sits on actual anatomy?
    # Anatomy = the largest bright connected region (the head), holes filled.
    head = (gray > max(10, 0.10 * float(gray.max()))).astype(np.uint8)
    if head.any():
        n_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(head, connectivity=8)
        if n_h > 1:
            biggest = 1 + int(np.argmax(h_stats[1:, cv2.CC_STAT_AREA]))
            head = (h_labels == biggest).astype(np.uint8)
        # Fill interior holes so ventricles/CSF are not treated as "outside".
        filled = head.copy()
        ff = (1 - head).astype(np.uint8)
        n_f, f_labels, f_stats, _ = cv2.connectedComponentsWithStats(ff, connectivity=4)
        for idx in range(1, n_f):
            x, y, w, h, _ = f_stats[idx]
            if x == 0 or y == 0 or x + w >= head.shape[1] or y + h >= head.shape[0]:
                continue
            filled[f_labels == idx] = 1
        head = filled
    result["inside_frac"] = float((binary & head).sum() / fg)
    result["align_ok"] = bool(result["inside_frac"] >= align_min)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    source: str
    n_stems: int
    orphans: dict[str, list[str]]
    geometry: dict[str, Any]
    duplicates: dict[str, Any]
    near_duplicates: dict[str, Any]
    group_stats: dict[str, Any]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_stems": self.n_stems,
            "orphans": self.orphans,
            "geometry": self.geometry,
            "duplicates": self.duplicates,
            "near_duplicates": self.near_duplicates,
            "group_stats": self.group_stats,
            "problems": self.problems,
            "notes": self.notes,
        }


def audit_source(
    spec: SourceSpec,
    *,
    workers: int = 2,
    align_min_inside: float = 0.90,
    phash_hamming_max: int = 3,
    phash_sample: int = 4000,
    seed: int = 42,
) -> AuditResult:
    """Run the full hygiene pass over one source."""
    stems = list_stems(spec)
    orphan_map = orphans(spec)

    logger.info("[%s] %d paired stems", spec.name, len(stems))
    if orphan_map["images_without_mask"] or orphan_map["masks_without_image"]:
        logger.warning(
            "[%s] orphans: %d images without mask, %d masks without image",
            spec.name,
            len(orphan_map["images_without_mask"]),
            len(orphan_map["masks_without_image"]),
        )

    image_paths = [resolve_image_path(spec, s) for s in stems]
    mask_paths = [resolve_mask_path(spec, s) for s in stems]

    # ── geometry + alignment + morphology ────────────────────────────────
    jobs = list(zip(image_paths, mask_paths, [align_min_inside] * len(stems)))
    with Pool(workers) as pool:
        geom = pool.map(_geom, jobs, chunksize=32)

    unreadable = [stems[i] for i, g in enumerate(geom) if g.get("unreadable")]
    shape_mismatch = [stems[i] for i, g in enumerate(geom) if g.get("shape_match") is False]
    non_binary = [stems[i] for i, g in enumerate(geom) if g.get("is_binary") is False]
    empty = [stems[i] for i, g in enumerate(geom) if g.get("empty")]
    full = [stems[i] for i, g in enumerate(geom) if g.get("full")]
    misaligned = [
        stems[i] for i, g in enumerate(geom)
        if not g.get("unreadable") and not g.get("empty") and not g.get("align_ok", True)
    ]

    fg_fracs = np.array([g["fg_frac"] for g in geom if not g.get("unreadable")], dtype=np.float64)
    inside = np.array(
        [g.get("inside_frac", 0.0) for g in geom if not g.get("unreadable") and not g.get("empty")],
        dtype=np.float64,
    )
    comps = np.array([g.get("n_components", 0) for g in geom if not g.get("unreadable")], dtype=np.int64)
    holes = np.array([g.get("n_holes", 0) for g in geom if not g.get("unreadable")], dtype=np.int64)
    solidity = np.array([g.get("solidity", 0.0) for g in geom if not g.get("unreadable")])
    circularity = np.array([g.get("circularity", 0.0) for g in geom if not g.get("unreadable")])
    resolutions = Counter(tuple(g["image_hw"]) for g in geom if not g.get("unreadable"))

    geometry = {
        "unreadable": unreadable,
        "shape_mismatch": shape_mismatch,
        "non_binary": non_binary,
        "empty_masks": empty,
        "full_masks": full,
        "misaligned": misaligned,
        "misaligned_count": len(misaligned),
        "fg_frac": {
            "min": float(fg_fracs.min()) if fg_fracs.size else None,
            "median": float(np.median(fg_fracs)) if fg_fracs.size else None,
            "max": float(fg_fracs.max()) if fg_fracs.size else None,
        },
        "inside_frac": {
            "min": float(inside.min()) if inside.size else None,
            "median": float(np.median(inside)) if inside.size else None,
        },
        "n_components": {
            "median": float(np.median(comps)) if comps.size else None,
            "max": int(comps.max()) if comps.size else None,
            "fragmented": int((comps > 2).sum()) if comps.size else 0,
        },
        "n_holes": {
            "total": int(holes.sum()) if holes.size else 0,
            "files_with_holes": int((holes > 0).sum()) if holes.size else 0,
        },
        "solidity_median": float(np.median(solidity)) if solidity.size else None,
        "circularity_median": float(np.median(circularity)) if circularity.size else None,
        "resolutions": {f"{h}x{w}": n for (h, w), n in resolutions.most_common()},
    }

    logger.info(
        "[%s] geometry: unreadable=%d shape_mismatch=%d non_binary=%d empty=%d full=%d misaligned=%d",
        spec.name, len(unreadable), len(shape_mismatch), len(non_binary),
        len(empty), len(full), len(misaligned),
    )
    if geometry["fg_frac"]["median"] is not None:
        logger.info(
            "[%s] fg_frac min=%.4f median=%.4f max=%.4f | inside_frac min=%.3f median=%.3f",
            spec.name, geometry["fg_frac"]["min"], geometry["fg_frac"]["median"],
            geometry["fg_frac"]["max"], geometry["inside_frac"]["min"], geometry["inside_frac"]["median"],
        )

    # ── exact duplicates ─────────────────────────────────────────────────
    with Pool(workers) as pool:
        image_hashes = pool.map(_sha256, image_paths, chunksize=64)
        mask_hashes = pool.map(_sha256, mask_paths, chunksize=64)

    by_image_hash: dict[str, list[int]] = defaultdict(list)
    for idx, digest in enumerate(image_hashes):
        if digest:
            by_image_hash[digest].append(idx)

    dup_groups = {h: idxs for h, idxs in by_image_hash.items() if len(idxs) > 1}
    same_mask_groups = 0
    conflicting_groups = 0
    conflicting_files = 0
    for digest, idxs in dup_groups.items():
        mask_set = {mask_hashes[i] for i in idxs}
        if len(mask_set) == 1:
            same_mask_groups += 1
        else:
            conflicting_groups += 1
            conflicting_files += len(idxs)

    mask_hash_groups: dict[str, list[int]] = defaultdict(list)
    for idx, digest in enumerate(mask_hashes):
        if digest:
            mask_hash_groups[digest].append(idx)
    mask_dup_groups = {h: idxs for h, idxs in mask_hash_groups.items() if len(idxs) > 1}

    duplicates = {
        "image_sha256_unique": len(by_image_hash),
        "mask_sha256_unique": len(set(h for h in mask_hashes if h)),
        "image_duplicate_groups": len(dup_groups),
        "image_duplicate_files": int(sum(len(v) for v in dup_groups.values())),
        "image_dup_groups_same_mask": same_mask_groups,
        "image_dup_groups_conflicting_mask": conflicting_groups,
        "image_dup_files_conflicting_mask": conflicting_files,
        "mask_duplicate_groups": len(mask_dup_groups),
        "cross_hash_collisions": len(set(h for h in image_hashes if h) & set(h for h in mask_hashes if h)),
    }
    logger.info(
        "[%s] sha256: %d duplicate image groups (%d files), of which %d carry a "
        "DIFFERENT mask covering %d files",
        spec.name, duplicates["image_duplicate_groups"], duplicates["image_duplicate_files"],
        conflicting_groups, conflicting_files,
    )

    # ── near duplicates (pHash) ──────────────────────────────────────────
    near_duplicates = _phash_pass(
        stems, image_paths, spec, hamming_max=phash_hamming_max,
        sample=phash_sample, seed=seed, workers=workers,
    )

    # ── grouping ─────────────────────────────────────────────────────────
    group_stats: dict[str, Any] = {"kind": spec.group_kind, "is_subject_verified": spec.group_is_subject_verified}
    if spec.group_of is not None:
        groups: dict[str, list[str]] = defaultdict(list)
        for stem in stems:
            groups[str(spec.group_of(stem))].append(stem)
        sizes = np.array([len(v) for v in groups.values()])
        group_stats.update(
            {
                "n_groups": len(groups),
                "slices_per_group": {
                    "median": float(np.median(sizes)),
                    "mean": float(sizes.mean()),
                    "max": int(sizes.max()),
                },
                "singleton_groups": int((sizes == 1).sum()),
            }
        )
        logger.info(
            "[%s] grouping: %d groups, median %.0f slices/group, max %d, %d singletons",
            spec.name, len(groups), float(np.median(sizes)), int(sizes.max()), int((sizes == 1).sum()),
        )
    else:
        group_stats.update({"n_groups": None, "note": "no recoverable group key; split uses contiguous blocks"})
        logger.warning("[%s] no group key available — split will use contiguous blocks", spec.name)

    # ── verdict ──────────────────────────────────────────────────────────
    problems: list[str] = []
    if orphan_map["images_without_mask"] or orphan_map["masks_without_image"]:
        problems.append("orphaned files present")
    if unreadable:
        problems.append(f"{len(unreadable)} unreadable files")
    if shape_mismatch:
        problems.append(f"{len(shape_mismatch)} image/mask shape mismatches")
    if non_binary:
        problems.append(f"{len(non_binary)} non-binary masks")
    if empty:
        problems.append(f"{len(empty)} empty masks")
    if full:
        problems.append(f"{len(full)} full-frame masks")
    if misaligned:
        problems.append(f"{len(misaligned)} masks below {align_min_inside:.0%} on anatomy")
    if conflicting_groups:
        problems.append(
            f"{conflicting_files} files are byte-identical images carrying different masks"
        )

    notes: list[str] = []
    if spec.group_kind == "lesion":
        notes.append(
            "Group key is derived from filenames (sequence and slice number stripped). "
            "It groups slices of the same lesion. Whether a lesion equals a patient is "
            "not determinable from the files — reported as a lesion key, never a patient key."
        )
    if spec.group_of is None:
        notes.append(
            "No group key exists for this source. Splits use contiguous index blocks; "
            "scan boundaries are not known, so a small number of slices at each block "
            "boundary may share a scan with the neighbouring split. Quantified in the report."
        )

    return AuditResult(
        source=spec.name,
        n_stems=len(stems),
        orphans=orphan_map,
        geometry=geometry,
        duplicates=duplicates,
        near_duplicates=near_duplicates,
        group_stats=group_stats,
        problems=problems,
        notes=notes,
    )


def _phash_pass(
    stems: list[str],
    image_paths: list[str],
    spec: SourceSpec,
    *,
    hamming_max: int,
    sample: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    """Perceptual near-duplicate detection with a same-group / cross-group split.

    The distinction matters: within-group hits are adjacent slices of one
    lesion (expected, harmless), cross-group hits are the ones that would
    indicate a duplicated image mislabelled as a different case.
    """
    import random

    rng = random.Random(seed)
    if sample and sample < len(stems):
        idxs = sorted(rng.sample(range(len(stems)), sample))
    else:
        idxs = list(range(len(stems)))

    with Pool(workers) as pool:
        hashes = pool.map(_phash, [(image_paths[i], 8, 4) for i in idxs], chunksize=64)

    keep = [(stems[idxs[k]], hashes[k]) for k in range(len(idxs)) if hashes[k] is not None]
    names = [k[0] for k in keep]
    values = np.array([k[1] for k in keep], dtype=np.uint64)
    n = len(names)

    counts = {f"hamming<={t}": {"within_group": 0, "cross_group": 0} for t in range(hamming_max + 1)}
    examples: list[dict[str, Any]] = []
    chunk = 512
    for start in range(0, n, chunk):
        block = values[start:start + chunk]
        dist = _popcount(block[:, None] ^ values[None, :])
        rows, cols = np.nonzero(dist <= hamming_max)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i >= j:
                continue
            d = int(dist[r, c])
            if spec.group_of is not None:
                same = spec.group_of(names[i]) == spec.group_of(names[j])
            else:
                # Without a group key, treat consecutive indices as the same scan.
                try:
                    same = abs(int(names[i]) - int(names[j])) <= 4
                except ValueError:
                    same = False
            for t in range(d, hamming_max + 1):
                key = f"hamming<={t}"
                counts[key]["within_group" if same else "cross_group"] += 1
            if not same and len(examples) < 20:
                examples.append({"a": names[i], "b": names[j], "hamming": d})

    logger.info("[%s] pHash over %d images:", spec.name, n)
    for key in sorted(counts):
        entry = counts[key]
        logger.info(
            "    %-12s within-group=%6d  cross-group=%6d",
            key, entry["within_group"], entry["cross_group"],
        )

    return {
        "sampled": n,
        "sampled_from": len(stems),
        "hamming_max": hamming_max,
        "counts": counts,
        "cross_group_examples": examples,
        "note": (
            "Threshold calibrated empirically on these datasets: the cross-group hit "
            "rate is flat up to 3 and rises sharply beyond it, which is where pHash "
            "starts matching shared composition rather than content."
        ),
    }


def cross_source_overlap(
    spec_a: SourceSpec,
    spec_b: SourceSpec,
    *,
    workers: int = 2,
    hamming_max: int = 3,
    sample: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    """Check whether two sources share any image.

    Exact bytes can never match across a PNG-grayscale / JPEG-RGB boundary, so
    the perceptual pass is the one that carries information here. Every
    candidate is then confirmed at pixel level — a perceptual hit alone is a
    hypothesis, not a finding.
    """
    import random

    stems_a = list_stems(spec_a)
    stems_b = list_stems(spec_b)
    paths_a = [resolve_image_path(spec_a, s) for s in stems_a]
    paths_b = [resolve_image_path(spec_b, s) for s in stems_b]

    with Pool(workers) as pool:
        sha_a = set(h for h in pool.map(_sha256, paths_a, chunksize=64) if h)
        sha_b = set(h for h in pool.map(_sha256, paths_b, chunksize=64) if h)

    rng = random.Random(seed)
    idx_a = sorted(rng.sample(range(len(stems_a)), min(sample, len(stems_a)))) if sample else list(range(len(stems_a)))
    idx_b = sorted(rng.sample(range(len(stems_b)), min(sample, len(stems_b)))) if sample else list(range(len(stems_b)))

    with Pool(workers) as pool:
        ha = pool.map(_phash, [(paths_a[i], 8, 4) for i in idx_a], chunksize=64)
        hb = pool.map(_phash, [(paths_b[i], 8, 4) for i in idx_b], chunksize=64)

    ka = [(stems_a[idx_a[k]], ha[k]) for k in range(len(idx_a)) if ha[k] is not None]
    kb = [(stems_b[idx_b[k]], hb[k]) for k in range(len(idx_b)) if hb[k] is not None]

    if not ka or not kb:
        return {"exact_sha256_overlap": len(sha_a & sha_b), "candidates": [], "confirmed": 0}

    va = np.array([k[1] for k in ka], dtype=np.uint64)
    vb = np.array([k[1] for k in kb], dtype=np.uint64)
    names_a = [k[0] for k in ka]
    names_b = [k[0] for k in kb]

    candidates: list[dict[str, Any]] = []
    chunk = 512
    for start in range(0, len(va), chunk):
        dist = _popcount(va[start:start + chunk, None] ^ vb[None, :])
        rows, cols = np.nonzero(dist <= hamming_max)
        for r, c in zip(rows, cols):
            candidates.append(
                {"a": names_a[start + int(r)], "b": names_b[int(c)], "hamming": int(dist[r, c])}
            )
            if len(candidates) >= 60:
                break
        if len(candidates) >= 60:
            break

    # Pixel confirmation: a genuine shared slice shows near-zero MAE.
    confirmed = []
    for cand in candidates:
        try:
            ia = cv2.imread(resolve_image_path(spec_a, cand["a"]), cv2.IMREAD_GRAYSCALE)
            ib = cv2.imread(resolve_image_path(spec_b, cand["b"]), cv2.IMREAD_GRAYSCALE)
        except FileNotFoundError:
            continue
        if ia is None or ib is None:
            continue
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]), interpolation=cv2.INTER_AREA)
        mae = float(np.abs(ia.astype(np.int16) - ib.astype(np.int16)).mean())
        entry = dict(cand)
        entry["pixel_mae"] = mae
        entry["same_image"] = mae < 3.0
        confirmed.append(entry)

    return {
        "exact_sha256_overlap": len(sha_a & sha_b),
        "exact_note": (
            "Expected to be 0 when the two sources store different container formats "
            "(PNG-grayscale vs JPEG-RGB); identical bytes are then impossible."
        ),
        "sampled": {"a": len(ka), "b": len(kb)},
        "candidates": confirmed,
        "confirmed_shared_images": sum(1 for c in confirmed if c["same_image"]),
    }
