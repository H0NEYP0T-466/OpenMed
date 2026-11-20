"""Array cache, dataset and batch sampler.

Cache
    Every surviving image goes through the *shared* preprocessing contract once
    (``preprocessor.to_canvas`` / ``mask_to_canvas``) and lands in a uint8 memmap.
    Masks are bit-packed (a lung mask is 8 KB instead of 64 KB). Infection masks are
    stored only for records that have one. ``verify_cache`` then decodes a sample of
    raw files through the *serving* path and requires byte-equality with the cache.

Dataset
    Returns the full frame and lung mask for every sample. For head-B-supervised
    samples it also returns a crop around the (augmented) lung field and the matching
    opacity target, because a 130 px lesion is 0.2 % of the frame and one pixel of
    drift destroys Dice. Unsupervised samples return zeros there and ``sup = False``.

Sampler
    Builds every rank's batches from ``(seed, epoch)`` alone, so ranks agree without
    communication, and guarantees ``min_supervised_per_batch`` supervised samples per
    batch. Without that head B starves and DDP raises on unused parameters.
    ``normal_neg_weight > 1`` oversamples supervised Normal (the true negatives);
    that is the fix for Normal false positives. Re-including Non-COVID never is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..labels import CLASS_IDS, CLASS_NORMAL
from ..preprocessor import (
    IMAGE_SIZE,
    PREPROCESS_VERSION,
    crop_resize_image,
    crop_resize_mask,
    decode_gray,
    full_box,
    lung_crop_box,
    mask_to_canvas,
    read_gray,
    read_mask,
    to_canvas,
)
from .augment import augment, jittered_crop_box
from .catalog import Record
from .config import TrainConfig
from .hygiene import CheckLog

logger = logging.getLogger("data")

PACKED = IMAGE_SIZE * IMAGE_SIZE // 8


def _fingerprint(records: list[Record]) -> str:
    digest = hashlib.sha256()
    digest.update(f"v{PREPROCESS_VERSION}|{IMAGE_SIZE}".encode())
    for r in records:
        digest.update(f"{r.uid}|{r.image_path}|{r.lung_mask_path}|{r.infection_mask_path}\n".encode())
    return digest.hexdigest()[:20]


def _load_one(task: tuple[int, str, str, str | None]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None]:
    row, image_path, lung_path, inf_path = task
    canvas, _ = to_canvas(read_gray(image_path), IMAGE_SIZE)
    lung = mask_to_canvas(read_mask(lung_path), IMAGE_SIZE)
    inf = mask_to_canvas(read_mask(inf_path), IMAGE_SIZE) if inf_path else None
    return row, canvas, np.packbits(lung.reshape(-1)), (
        np.packbits(inf.reshape(-1)) if inf is not None else None)


class ArrayCache:
    """Read access to the memmaps. Opened lazily so DataLoader workers get their own handles."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.n = int(self.meta["n"])
        self.row_of_uid: dict[int, int] = {int(k): v for k, v in self.meta["row_of_uid"].items()}
        self.inf_row_of_uid: dict[int, int] = {int(k): v for k, v in self.meta["inf_row_of_uid"].items()}
        self._images: np.memmap | None = None
        self._lung: np.memmap | None = None
        self._inf: np.memmap | None = None

    def _open(self) -> None:
        if self._images is not None:
            return
        d = self.directory
        self._images = np.memmap(os.path.join(d, "images.u8"), dtype=np.uint8, mode="r",
                                 shape=(self.n, IMAGE_SIZE, IMAGE_SIZE))
        self._lung = np.memmap(os.path.join(d, "lung.bits"), dtype=np.uint8, mode="r",
                               shape=(self.n, PACKED))
        k = len(self.inf_row_of_uid)
        self._inf = (np.memmap(os.path.join(d, "inf.bits"), dtype=np.uint8, mode="r",
                               shape=(k, PACKED)) if k else None)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_images"] = state["_lung"] = state["_inf"] = None
        return state

    def image(self, uid: int) -> np.ndarray:
        self._open()
        assert self._images is not None
        return np.array(self._images[self.row_of_uid[uid]])

    def lung(self, uid: int) -> np.ndarray:
        self._open()
        assert self._lung is not None
        bits = np.unpackbits(np.array(self._lung[self.row_of_uid[uid]]))
        return bits.reshape(IMAGE_SIZE, IMAGE_SIZE)

    def infection(self, uid: int) -> np.ndarray | None:
        self._open()
        row = self.inf_row_of_uid.get(uid)
        if row is None or self._inf is None:
            return None
        return np.unpackbits(np.array(self._inf[row])).reshape(IMAGE_SIZE, IMAGE_SIZE)


def build_cache(
    records: list[Record], directory: str, *, workers: int, force: bool = False
) -> ArrayCache:
    """Build (or reuse, when the fingerprint matches) the uint8 memmap cache."""
    os.makedirs(directory, exist_ok=True)
    meta_path = os.path.join(directory, "meta.json")
    fingerprint = _fingerprint(records)
    if not force and os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("fingerprint") == fingerprint and meta.get("complete"):
            logger.info("cache reused: %s (%d images, fingerprint %s)", directory, meta["n"], fingerprint)
            return ArrayCache(directory)

    n = len(records)
    with_inf = [r for r in records if r.infection_mask_path and r.in_infection_tree]
    inf_row = {r.uid: i for i, r in enumerate(with_inf)}
    row_of_uid = {r.uid: i for i, r in enumerate(records)}
    logger.info("building cache: %d images, %d with infection masks -> %s", n, len(with_inf), directory)

    images = np.memmap(os.path.join(directory, "images.u8"), dtype=np.uint8, mode="w+",
                       shape=(n, IMAGE_SIZE, IMAGE_SIZE))
    lung = np.memmap(os.path.join(directory, "lung.bits"), dtype=np.uint8, mode="w+", shape=(n, PACKED))
    inf = (np.memmap(os.path.join(directory, "inf.bits"), dtype=np.uint8, mode="w+",
                     shape=(len(with_inf), PACKED)) if with_inf else None)

    tasks = [(row_of_uid[r.uid], r.image_path, r.lung_mask_path,
              r.infection_mask_path if r.uid in inf_row else None) for r in records]
    pool = Pool(max(1, workers)) if workers > 1 and n >= 64 else None
    iterator = pool.imap_unordered(_load_one, tasks, chunksize=32) if pool else map(_load_one, tasks)
    try:
        for done, (row, canvas, lung_bits, inf_bits) in enumerate(iterator, start=1):
            images[row] = canvas
            lung[row] = lung_bits
            if inf_bits is not None and inf is not None:
                uid = records[row].uid
                inf[inf_row[uid]] = inf_bits
            if done % 5000 == 0:
                logger.info("  cached %d / %d", done, n)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    images.flush()
    lung.flush()
    if inf is not None:
        inf.flush()
    del images, lung, inf

    meta = {"fingerprint": fingerprint, "n": n, "complete": True, "image_size": IMAGE_SIZE,
            "preprocess_version": PREPROCESS_VERSION,
            "row_of_uid": {str(k): v for k, v in row_of_uid.items()},
            "inf_row_of_uid": {str(k): v for k, v in inf_row.items()}}
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    os.replace(tmp, meta_path)
    return ArrayCache(directory)


def verify_cache(cache: ArrayCache, records: list[Record], log: CheckLog, n: int = 96, seed: int = 0) -> None:
    """Raw bytes -> *serving* decode and canvas must equal the cache; masks must be binary."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(records), size=min(n, len(records)), replace=False)
    image_bad = mask_bad = inf_bad = 0
    for i in pick:
        r = records[int(i)]
        with open(r.image_path, "rb") as handle:
            canvas, _ = to_canvas(decode_gray(handle.read()), IMAGE_SIZE)
        if not np.array_equal(canvas, cache.image(r.uid)):
            image_bad += 1
        if not np.array_equal(mask_to_canvas(read_mask(r.lung_mask_path), IMAGE_SIZE), cache.lung(r.uid)):
            mask_bad += 1
        cached_inf = cache.infection(r.uid)
        if r.infection_mask_path:
            expected = mask_to_canvas(read_mask(r.infection_mask_path), IMAGE_SIZE)
            if cached_inf is None or not np.array_equal(expected, cached_inf):
                inf_bad += 1
    status = "PASS" if not (image_bad or mask_bad or inf_bad) else "FAIL"
    log.add("H19", "cache integrity against raw bytes", status,
            f"{len(pick)} files decoded from raw bytes through the serving path: {image_bad} image, "
            f"{mask_bad} lung-mask, {inf_bad} infection-mask difference(s) from what training sees")
    if status == "FAIL":
        raise RuntimeError("Cached arrays do not reproduce the serving preprocessing.")


# ------------------------------------------------------------------------ dataset


class SegDataset(Dataset):
    def __init__(self, cache: ArrayCache, records: list[Record], *, train: bool, cfg: TrainConfig,
                 want_crops: bool) -> None:
        self.cache = cache
        self.records = records
        self.train = train
        self.cfg = cfg
        self.want_crops = want_crops
        self.epoch = 0
        self.sup = np.array([r.opacity_supervised for r in records], dtype=bool)
        self.cls = np.array([CLASS_IDS[r.cls] for r in records], dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self.cache.image(record.uid)
        lung = self.cache.lung(record.uid)
        supervised = bool(self.sup[index]) and self.want_crops
        opacity = self.cache.infection(record.uid) if supervised else None
        if supervised and opacity is None:
            # Provenance says supervised but no mask exists: an indexing bug, not data.
            raise RuntimeError(f"{record.filename} is flagged opacity-supervised but has no infection mask")

        if self.train and self.cfg.aug_enabled:
            rng = np.random.default_rng([self.cfg.seed, self.epoch, index])
            image, lung, opacity = augment(image, lung, opacity, rng, self.cfg)
        else:
            rng = None

        crop_img = np.zeros_like(image)
        crop_msk = np.zeros_like(image)
        if supervised and opacity is not None:
            if rng is not None:
                box = jittered_crop_box(lung, rng, self.cfg)
            else:
                box = lung_crop_box(lung) or full_box(lung.shape)
            crop_img = crop_resize_image(image, box)
            crop_msk = crop_resize_mask(opacity, box)
        return {
            "full": torch.from_numpy(np.ascontiguousarray(image)),
            "lung": torch.from_numpy(np.ascontiguousarray(lung)),
            "crop": torch.from_numpy(crop_img),
            "opacity": torch.from_numpy(crop_msk),
            "sup": torch.tensor(supervised),
            "cls": torch.tensor(int(self.cls[index])),
            "uid": torch.tensor(record.uid),
        }


def collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


# ------------------------------------------------------------------------ sampler


class EpochBatchSampler:
    """Deterministic per-epoch batches, identical on every rank, sliced by rank."""

    def __init__(self, sup: np.ndarray, cls: np.ndarray, *, batch_size: int, world: int, rank: int,
                 seed: int, min_supervised: int, normal_neg_weight: float) -> None:
        self.sup, self.cls = sup, cls
        self.batch_size, self.world, self.rank = batch_size, world, rank
        self.seed = seed
        self.min_supervised = min_supervised
        self.neg_weight = normal_neg_weight
        self.epoch = 0
        self.sup_pool = np.flatnonzero(sup)
        self.normal_pool = np.flatnonzero(sup & (cls == CLASS_IDS[CLASS_NORMAL]))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _all_batches(self) -> list[np.ndarray]:
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(len(self.sup))
        if self.neg_weight > 1.0 and self.normal_pool.size:
            extra = rng.choice(self.normal_pool, size=round((self.neg_weight - 1.0) * self.normal_pool.size))
            order = rng.permutation(np.concatenate([order, extra]))
        usable = (len(order) // (self.batch_size * self.world)) * self.batch_size * self.world
        batches = list(order[:usable].reshape(-1, self.batch_size))
        if self.min_supervised > 0 and self.sup_pool.size:
            need = min(self.min_supervised, self.batch_size)
            for batch in batches:
                flags = self.sup[batch]
                short = need - int(flags.sum())
                if short <= 0:
                    continue
                slots = rng.choice(np.flatnonzero(~flags), size=short, replace=False)
                batch[slots] = rng.choice(self.sup_pool, size=short, replace=short > self.sup_pool.size)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._all_batches()[self.rank::self.world]:
            yield [int(i) for i in batch]

    def __len__(self) -> int:
        usable = (len(self.sup) // (self.batch_size * self.world)) * self.world
        if self.neg_weight > 1.0:
            usable = ((len(self.sup) + round((self.neg_weight - 1.0) * self.normal_pool.size))
                      // (self.batch_size * self.world)) * self.world
        return usable // self.world
