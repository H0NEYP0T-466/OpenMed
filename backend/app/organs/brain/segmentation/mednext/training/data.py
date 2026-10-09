"""Array cache and Dataset.

Decoding 12k JPEG/PNG pairs every epoch would starve the GPUs, so every record
is decoded and resized *once* by the shared preprocessor into two uint8 arrays
(``images.npy``, ``masks.npy``) that all ranks and loader workers memory-map.
Augmentation and normalisation then run per sample on the 256x256 canvas.

Building the cache is also a hygiene stage (H15): it measures what resizing does
to the masks, drops lesions that vanish at model resolution, and re-derives a
random sample straight from the files to prove the cache matches them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from multiprocessing import Pool
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocessor import (
    PREPROCESS_VERSION,
    mask_to_canvas,
    normalise,
    read_gray,
    to_canvas,
)
from .audit import CheckLog
from .augment import augment
from .splits import Record, SplitManifest

logger = logging.getLogger("data")

SOURCE_IDS = {"openmed": 0, "btsc": 1}


def _build_one(args: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray, float]:
    cv2.setNumThreads(1)
    image_path, mask_path, size = args
    gray = read_gray(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Mask could not be read: {mask_path}")
    canvas, _ = to_canvas(gray, size)
    mask_canvas = mask_to_canvas(mask, size)
    plane = mask[:, :, 0] if mask.ndim == 3 else mask
    return canvas, mask_canvas, float((plane >= 128).mean())


class ArrayCache:
    """Memory-mapped ``(N, S, S)`` uint8 image and mask arrays, plus row lookup."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        with open(os.path.join(directory, "meta.json")) as handle:
            self.meta = json.load(handle)
        self.size = int(self.meta["size"])
        self.index = {(s, t): i for i, (s, t) in enumerate(zip(self.meta["sources"], self.meta["stems"], strict=True))}
        self._images: Optional[np.ndarray] = None
        self._masks: Optional[np.ndarray] = None

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(os.path.join(self.directory, "images.npy"), mmap_mode="r")
        return self._images

    @property
    def masks(self) -> np.ndarray:
        if self._masks is None:
            self._masks = np.load(os.path.join(self.directory, "masks.npy"), mmap_mode="r")
        return self._masks

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_images"] = None
        state["_masks"] = None
        return state

    def row(self, record: Record) -> int:
        return self.index[(record.source, record.stem)]


def _cache_digest(records: list[Record], size: int) -> str:
    payload = json.dumps(
        [(r.source, r.stem, r.image_path) for r in records] + [size, PREPROCESS_VERSION]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_cache(
    records: list[Record],
    directory: str,
    size: int,
    workers: int,
    log: CheckLog,
    seed: int = 42,
    force: bool = False,
) -> ArrayCache:
    """Build (or reuse) the cache for ``records`` and run the H15 checks."""
    os.makedirs(directory, exist_ok=True)
    digest = _cache_digest(records, size)
    meta_path = os.path.join(directory, "meta.json")

    if not force and os.path.isfile(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
        if meta.get("digest") == digest:
            logger.info("reusing array cache %s (%d rows)", directory, len(meta["stems"]))
            cache = ArrayCache(directory)
            verify_cache(cache, records, log, seed)
            return cache

    n = len(records)
    images = np.lib.format.open_memmap(
        os.path.join(directory, "images.npy"), mode="w+", dtype=np.uint8, shape=(n, size, size)
    )
    masks = np.lib.format.open_memmap(
        os.path.join(directory, "masks.npy"), mode="w+", dtype=np.uint8, shape=(n, size, size)
    )
    jobs = [(r.image_path, r.mask_path, size) for r in records]
    original_fg = np.zeros(n, dtype=np.float64)
    step = max(1, n // 10)
    logger.info("building array cache: %d records at %dx%d", n, size, size)
    with Pool(max(1, workers)) as pool:
        for i, (canvas, mask_canvas, fg) in enumerate(pool.imap(_build_one, jobs, chunksize=32)):
            images[i], masks[i], original_fg[i] = canvas, mask_canvas, fg
            if (i + 1) % step == 0:
                logger.info("  cached %d / %d", i + 1, n)
    images.flush()
    masks.flush()

    meta = {
        "digest": digest, "size": size, "preprocess_version": PREPROCESS_VERSION,
        "sources": [r.source for r in records], "stems": [r.stem for r in records],
        "original_fg_frac": original_fg.tolist(),
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle)
    cache = ArrayCache(directory)
    verify_cache(cache, records, log, seed)
    return cache


def verify_cache(cache: ArrayCache, records: list[Record], log: CheckLog, seed: int) -> list[Record]:
    """H15/H16: resolution survival, value domain and a from-file re-derivation."""
    images, masks = cache.images, cache.masks
    fg = np.array([int(masks[cache.row(r)].sum()) for r in records])
    vanished = [records[i] for i in np.nonzero(fg == 0)[0]]
    small = int((fg < 16).sum())
    original = np.array([cache.meta["original_fg_frac"][cache.row(r)] for r in records])
    model_frac = fg / float(cache.size**2)
    drift = np.abs(model_frac - original) / np.maximum(original, 1e-9)

    values_ok = bool(np.isin(np.unique(masks[: min(len(masks), 512)]), (0, 1)).all())
    log.add("H15", "cache resolution", "PASS" if not vanished else "WARN",
            f"{len(vanished)} masks vanish at {cache.size}px, {small} keep < 16 px; "
            f"median relative area change {np.median(drift):.3%}, p99 {np.percentile(drift, 99):.3%}; "
            f"mask values in {{0,1}}: {values_ok}",
            vanished=[r.stem for r in vanished[:20]], area_drift_p99=float(np.percentile(drift, 99)))

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(records), size=min(64, len(records)), replace=False)
    mismatched = 0
    for i in picks:
        row = cache.row(records[i])
        canvas, mask_canvas, _ = _build_one((records[i].image_path, records[i].mask_path, cache.size))
        if not (np.array_equal(canvas, images[row]) and np.array_equal(mask_canvas, masks[row])):
            mismatched += 1
    log.add("H16", "cache vs files", "PASS" if not mismatched else "FAIL",
            f"{len(picks)} random rows re-derived from disk, {mismatched} differ")
    if mismatched:
        raise RuntimeError("Array cache does not match the source files; delete it and rebuild.")
    return vanished


def vanished_records(cache: ArrayCache, records: list[Record]) -> list[Record]:
    """Records whose mask has no foreground left at model resolution."""
    masks = cache.masks
    empty = np.array([not masks[cache.row(r)].any() for r in records])
    return [r for r, flag in zip(records, empty, strict=True) if flag]


def drop_vanished(manifest: SplitManifest, vanished: list[Record]) -> int:
    keys = {(r.source, r.stem) for r in vanished}
    removed = 0
    for name in ("train", "val", "test"):
        kept = [r for r in manifest.part(name) if (r.source, r.stem) not in keys]
        removed += len(manifest.part(name)) - len(kept)
        setattr(manifest, name, kept)
    return removed


class SegDataset(Dataset):
    """Cached slices -> ``(image, mask, source_id, row)`` tensors."""

    def __init__(
        self, cache: ArrayCache, records: list[Record], *, train: bool, cfg: Any, seed: int = 0
    ) -> None:
        self.cache = cache
        self.rows = np.array([cache.row(r) for r in records], dtype=np.int64)
        self.sources = np.array([SOURCE_IDS.get(r.source, 0) for r in records], dtype=np.int64)
        self.train = train
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        row = int(self.rows[index])
        image = np.array(self.cache.images[row])
        mask = np.array(self.cache.masks[row])
        if self.train and self.cfg.aug_enabled:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            image, mask = augment(image, mask, rng, self.cfg)
        x = torch.from_numpy(normalise(image)[None])
        y = torch.from_numpy(mask.astype(np.float32)[None])
        return x, y, int(self.sources[index]), index
