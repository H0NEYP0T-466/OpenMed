"""Brain tumour classification dataset, discovery and leakage-free splitting.

Supports two layouts:
  - manifest mode: DATA.json + Images_/ tree (the 42-class archive)
  - folder mode  : Images_/<class>/<files> with no manifest (flattened
    WHO-family dataset) - class label comes from the folder name.

Splitting is group-aware. Classes in `INDEPENDENT_CLASSES` hold one study per
file, so every file is its own group; all other classes hold patient slices,
so visually verified series are forced into a single split, and byte-identical
duplicates are collapsed everywhere. See `app/organs/brain/README.md`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SIGNATURE_SIZE = 32
CONTENT_SIZE = 64
SERIES_CORRELATION_THRESHOLD = 0.90
CACHE_FORMAT_VERSION = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Classes whose files are independent studies (one image = one subject); no
# slice-series grouping applies. Everything else is patient-slice based and
# gets GroupKFold-style whole-group splitting.
INDEPENDENT_CLASSES = frozenset({
    "Normal",
    "Pituitary",
    "Gliomas",
    "Meningothelial Tumors",
})


@dataclass(frozen=True)
class DatasetView:
    """Unified view over manifest and folder dataset layouts."""

    keys: list[str]                        # relative paths under images_base, sorted
    labels: dict[str, str]                 # key -> class name
    metadata: dict[str, dict[str, Any]]    # key -> per-sample metadata
    images_base: str
    cache_dir: str
    manifest_path: str | None = None       # DATA.json when present


def discover_dataset(data_root: str) -> DatasetView:
    """Discover a dataset in manifest or folder layout.

    Manifest mode is tried first so existing DATA.json archives keep their
    rich metadata; folder mode labels images by their immediate class folder.
    """
    try:
        json_path, images_base = BrainTumorDataset.locate_data_and_images(data_root)
    except FileNotFoundError:
        json_path = None

    if json_path is not None:
        with open(json_path) as handle:
            raw_data = json.load(handle)
        keys = sorted(
            path
            for path in raw_data
            if not path.endswith(BrainTumorDataset.MASK_SUFFIX)
        )
        metadata = {key: dict(raw_data[key]) for key in keys}
        labels = {key: metadata[key]["class"] for key in keys}
        return DatasetView(
            keys=keys,
            labels=labels,
            metadata=metadata,
            images_base=images_base,
            cache_dir=os.path.dirname(json_path),
            manifest_path=json_path,
        )

    canonical_set = {
        "Germ Cell Tumors",
        "Gliomas",
        "Medulloblastoma",
        "Meningothelial Tumors",
        "Mesenchymal (Non-Meningothelial Tumors)",
        "Mixed Neuronal and Neuronal-Glial Tumors",
        "Normal",
        "Pituitary",
        "Schwannoma",
    }
    base = None
    for candidate in (
        os.path.join(data_root, "archive", "Images_"),
        os.path.join(data_root, "Images_"),
        os.path.join(data_root, "archive"),
        data_root,
    ):
        if os.path.isdir(candidate):
            try:
                subdirs = {d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))}
                if len(subdirs & canonical_set) >= 2:
                    base = candidate
                    break
            except OSError:
                continue

    if base is None and os.path.isdir(data_root):
        for root, dirs, _ in os.walk(data_root):
            if len(set(dirs) & canonical_set) >= 2:
                base = root
                break

    if base is None:
        base = data_root

    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"No DATA.json and no image folders found under '{data_root}'."
        )

    keys: list[str] = []
    labels: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for dirpath, _, filenames in os.walk(base):
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            if name.endswith(BrainTumorDataset.MASK_SUFFIX):
                continue
            full = os.path.join(dirpath, name)
            key = os.path.relpath(full, base)
            parts = key.split(os.sep)
            if len(parts) < 2:
                logger.warning("Skipping image outside a class folder: %s", key)
                continue
            cls = parts[0]
            keys.append(key)
            labels[key] = cls
            metadata[key] = {
                "class": cls,
                "tumor_type": cls,
                "filename": name,
                "sequence": "",
                "location": [],
                "point": [],
                "has_lesion": 0 if cls == "Normal" else 1,
                "relative_path": key,
            }
    if not keys:
        raise FileNotFoundError(f"No class folders with images found under '{base}'.")
    keys.sort()
    cache_target = data_root if os.path.isdir(data_root) else base
    if not os.access(cache_target, os.W_OK):
        cache_target = os.environ.get("OPENMED_CACHE_DIR", tempfile.gettempdir())

    return DatasetView(
        keys=keys,
        labels=labels,
        metadata=metadata,
        images_base=base,
        cache_dir=cache_target,
        manifest_path=None,
    )


def _numeric_stem(relative_path: str) -> str:
    """Filename stem with a trailing numeric index removed, lowercased."""
    stem = os.path.splitext(os.path.basename(relative_path))[0]
    stripped = stem.rstrip("0123456789").rstrip(" _-")
    return stripped.strip().lower() or stem.lower()


def _sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _content_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _luminance_grid(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.uint8)


def _signature(path: str) -> list[float]:
    """Mean-removed, unit-normalised low-resolution luminance signature."""
    array = _luminance_grid(path, SIGNATURE_SIZE).astype(np.float32)
    array -= array.mean()
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array /= norm
    return array.reshape(-1).tolist()


def _probe_image(args: tuple[str, str]) -> tuple[str, str, list[float], str]:
    key, absolute = args
    try:
        content = _content_digest(_luminance_grid(absolute, CONTENT_SIZE))
        return key, _sha256(absolute), _signature(absolute), content
    except Exception as exc:  # noqa: BLE001 - recorded and surfaced by caller
        logger.warning("Unreadable image during grouping probe: %s (%s)", key, exc)
        return key, f"error:{exc}", [], ""


class BrainTumorDataset(Dataset):
    """Slice-level brain MRI classification dataset driven by a DATA.json manifest."""

    MASK_SUFFIX = "_mask.png"

    def __init__(
        self,
        data_root: str,
        transform: Any | None = None,
        split_indices: Sequence[int] | None = None,
        transforms_by_class: Mapping[str, Any] | None = None,
    ) -> None:
        self.view = discover_dataset(data_root)
        self.json_path = self.view.manifest_path or ""
        self.images_base = self.view.images_base
        self.data_root = self.view.cache_dir
        self.transform = transform
        self.transforms_by_class: dict[str, Any] = dict(transforms_by_class or {})

        self.class_names: list[str] = sorted(set(self.view.labels.values()))
        self.tumor_types: list[str] = sorted(
            {
                meta.get("tumor_type", meta["class"])
                for meta in self.view.metadata.values()
            }
        )
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        self.samples: list[dict[str, Any]] = [
            {
                "path": os.path.join(self.images_base, key),
                "relative_path": key,
                "metadata": self.view.metadata[key],
            }
            for key in self.view.keys
        ]

        if split_indices is not None:
            self.samples = [self.samples[i] for i in split_indices]

        logger.info(
            "Loaded dataset with %d samples. (%d classes%s)",
            len(self.samples),
            len(self.class_names),
            "" if self.view.manifest_path else ", folder layout",
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int, dict[str, Any]]:
        last_error: Exception | None = None
        for attempt, candidate in enumerate(self._candidate_indices(idx)):
            sample = self.samples[candidate]
            try:
                with Image.open(sample["path"]) as raw:
                    image = raw.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Image load failed for %s: %s", sample["path"], exc)
                continue

            label = self.class_to_idx[sample["metadata"]["class"]]
            transform = self.transforms_by_class.get(
                sample["metadata"]["class"], self.transform
            )
            if transform is not None:
                image = transform(image)

            metadata = sample["metadata"]
            meta_dict = {
                "location": metadata.get("location", []),
                "point": metadata.get("point", []),
                "sequence": metadata.get("sequence", ""),
                "tumor_type": metadata.get("tumor_type", ""),
                "filename": metadata.get("filename", ""),
                "has_lesion": metadata.get("has_lesion", 0),
                "relative_path": sample["relative_path"],
            }
            if attempt:
                logger.debug("Substituted readable sample for index %d", idx)
            return image, label, meta_dict

        raise RuntimeError(
            f"No readable image reachable from index {idx}"
        ) from last_error

    def _candidate_indices(self, idx: int, maximum: int = 8) -> Iterable[int]:
        total = len(self.samples)
        for offset in range(min(maximum, total)):
            yield (idx + offset) % total

    @staticmethod
    def collate_fn(
        batch: Sequence[tuple[Any, int, dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        metas = [item[2] for item in batch]
        return images, labels, metas

    @staticmethod
    def locate_data_and_images(data_root: str) -> tuple[str, str]:
        """Discover DATA.json and its image root across nested, zipped or Kaggle layouts."""
        json_path = None
        data_dir = data_root

        if os.path.isfile(data_root):
            if data_root.endswith(".json"):
                json_path = data_root
                data_dir = os.path.dirname(data_root)
            elif data_root.endswith(".zip"):
                data_dir = _safe_extract_zip(data_root)

        if json_path is None:
            for candidate in (
                os.path.join(data_dir, "DATA.json"),
                os.path.join(data_dir, "archive", "DATA.json"),
            ):
                if os.path.isfile(candidate):
                    json_path = candidate
                    data_dir = os.path.dirname(candidate)
                    break

        if json_path is None and os.path.isdir(data_root):
            for root, _, files in os.walk(data_root):
                if "DATA.json" in files:
                    json_path = os.path.join(root, "DATA.json")
                    data_dir = root
                    break

        if json_path is None:
            raise FileNotFoundError(
                f"Could not locate DATA.json inside '{data_root}'. Please verify the path."
            )

        images_base = data_dir
        for candidate in (
            os.path.join(data_dir, "Images_", "Images_"),
            os.path.join(data_dir, "Images_"),
            os.path.join(os.path.dirname(data_dir), "Images_", "Images_"),
            os.path.join(os.path.dirname(data_dir), "Images_"),
        ):
            if os.path.isdir(candidate):
                images_base = candidate
                break

        return json_path, images_base

    @staticmethod
    def compute_class_weights(
        dataset_samples: Sequence[dict[str, Any]],
        class_names: Sequence[str],
        smoothing: float = 1.0,
    ) -> torch.Tensor:
        """Inverse-frequency class weights, mean-normalised and clipped.

        `smoothing=1.0` is full 'balanced' weighting (sklearn convention):
        a class with 8x fewer samples gets an 8x larger loss weight.
        Lower values soften it (0.5 = square-root smoothing).
        """
        counts = Counter(sample["metadata"]["class"] for sample in dataset_samples)
        num_classes = len(class_names)
        total = len(dataset_samples) or 1

        weights = np.ones(num_classes, dtype=np.float32)
        for idx, name in enumerate(class_names):
            count = counts.get(name, 0)
            if count > 0:
                weights[idx] = (total / (num_classes * count)) ** smoothing

        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights /= mean_weight

        return torch.tensor(np.clip(weights, 0.2, 8.0), dtype=torch.float)

    @staticmethod
    def sample_weights(
        dataset_samples: Sequence[dict[str, Any]],
        class_names: Sequence[str],
        class_weights: torch.Tensor,
        smoothing: float = 0.5,
    ) -> list[float]:
        """Per-sample oversampling weights for WeightedRandomSampler.

        smoothing=1.0 fully balances sampling (rare images repeat ~8x per
        epoch, which invites memorisation); 0.5 is the sqrt compromise.
        """
        index_of = {name: idx for idx, name in enumerate(class_names)}
        values = class_weights.tolist()
        return [
            values[index_of[sample["metadata"]["class"]]] ** smoothing
            for sample in dataset_samples
        ]

    @staticmethod
    def build_group_ids(
        json_path: str | None,
        images_base: str,
        keys: Sequence[str],
        seed: int = 42,
        labels: Sequence[str] | None = None,
        cache_dir: str | None = None,
    ) -> list[str]:
        """Group ids for `keys`, one group per source subject (see `build_case_groups`)."""
        return build_case_groups(
            json_path=json_path,
            images_base=images_base,
            keys=keys,
            seed=seed,
            labels=labels,
            cache_dir=cache_dir,
        )

    @staticmethod
    def get_stratified_splits(
        data_root: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return leak-free train/val/test indices grouped by source subject."""
        view = discover_dataset(data_root)
        keys = view.keys
        labels = [view.labels[key] for key in keys]

        group_ids = build_case_groups(
            json_path=view.manifest_path,
            images_base=view.images_base,
            keys=keys,
            seed=seed,
            labels=labels,
            cache_dir=view.cache_dir,
        )

        targets = {
            "train": max(0.0, 1.0 - test_size - val_size),
            "val": val_size,
            "test": test_size,
        }
        assignment = _grouped_stratified_assign(group_ids, labels, targets, seed)

        buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for index in range(len(keys)):
            buckets[assignment[group_ids[index]]].append(index)

        _report_split_quality(keys, labels, group_ids, assignment, buckets)

        return buckets["train"], buckets["val"], buckets["test"]


def _safe_extract_zip(archive_path: str) -> str:
    """Extract a dataset archive, rejecting members that would escape the target directory."""
    destination = os.path.join(
        os.path.dirname(os.path.abspath(archive_path)), "unzipped_dataset"
    )
    marker = os.path.join(destination, ".extraction_complete")
    if os.path.isfile(marker):
        return destination

    os.makedirs(destination, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        root = os.path.realpath(destination)
        for member in archive.namelist():
            target = os.path.realpath(os.path.join(destination, member))
            if target != root and not target.startswith(root + os.sep):
                raise RuntimeError(f"Refusing unsafe zip member path: {member!r}")
        archive.extractall(destination)

    with open(marker, "w") as handle:
        handle.write(os.path.basename(archive_path))
    return destination


def _cache_path_for(cache_dir: str) -> str:
    return os.path.join(cache_dir, ".case_group_cache.json")


def _fingerprint(
    json_path: str | None, images_base: str, keys: Sequence[str]
) -> dict[str, Any]:
    """Identity for the exact image set the cache was derived from.

    Per-file size and mtime are hashed together rather than reduced to a max and
    a sum, so a single replaced frame cannot hide inside an unchanged total.
    Files that cannot be stat-ed are recorded explicitly instead of being
    silently dropped from the digest.
    """
    hasher = hashlib.sha256()
    unreadable = 0
    for key in keys:
        try:
            stat = os.stat(os.path.join(images_base, key))
        except OSError:
            unreadable += 1
            hasher.update(f"{key}|missing".encode())
            continue
        hasher.update(f"{key}|{stat.st_size}|{stat.st_mtime_ns}".encode())

    return {
        "manifest_mtime": os.path.getmtime(json_path) if json_path else 0.0,
        "image_digest": hasher.hexdigest(),
        "count": len(keys),
        "unreadable": unreadable,
    }


def _load_cache(
    cache_dir: str, keys: Sequence[str], fingerprint: dict[str, Any]
) -> dict[str, dict[str, str]] | None:
    path = _cache_path_for(cache_dir)
    if not os.path.isfile(path):
        fallback = os.path.join(tempfile.gettempdir(), ".case_group_cache.json")
        if os.path.isfile(fallback):
            path = fallback
        else:
            return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable case-group cache: %s", exc)
        return None

    if payload.get("format") != CACHE_FORMAT_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        logger.info("Case-group cache is stale for this image set; recomputing.")
        return None

    records = payload.get("records", {})
    if set(records) != set(keys):
        return None
    return records


def _probe_images(
    images_base: str, keys: Sequence[str], workers: int
) -> dict[str, dict[str, str]]:
    pairs = [(key, os.path.join(images_base, key)) for key in keys]
    results: dict[str, dict[str, str]] = {}

    if workers and workers > 1 and len(pairs) > 256:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            probed = list(executor.map(_probe_image, pairs, chunksize=64))
    else:
        probed = [_probe_image(pair) for pair in pairs]

    for key, digest, signature, content in probed:
        results[key] = {
            "sha256": digest,
            "content": content,
            "signature": base64_signature(signature),
        }
    return results


def base64_signature(values: Sequence[float]) -> str:
    if not values:
        return ""
    return base64.b64encode(
        np.asarray(values, dtype=np.float32).tobytes(order="C")
    ).decode("ascii")


def decode_signature(encoded: str) -> np.ndarray:
    if not encoded:
        return np.zeros(SIGNATURE_SIZE * SIGNATURE_SIZE, dtype=np.float32)
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


def build_case_groups(
    json_path: str | None,
    images_base: str,
    keys: Sequence[str],
    seed: int = 42,
    workers: int | None = None,
    labels: Sequence[str] | None = None,
    cache_dir: str | None = None,
) -> list[str]:
    """Map every image to a group id such that one group never spans two splits.

    Two images share a group when they are byte-identical, when their pixels
    are identical at 64x64, or when they share a numeric filename stem *within
    one class* and are measurably the same picture. Classes listed in
    `INDEPENDENT_CLASSES` hold one study per file and are never series-merged.
    The correlation test is what separates true slice series from unrelated
    scans that merely share a naming prefix.
    """
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    anchor_dir = cache_dir or (
        os.path.dirname(json_path) if json_path else images_base
    )

    fingerprint = _fingerprint(json_path, images_base, keys)
    records = _load_cache(anchor_dir, keys, fingerprint)
    if records is None:
        logger.info("Computing image signatures for leakage-safe grouping...")
        records = _probe_images(images_base, keys, workers)
        _write_cache(anchor_dir, keys, records, fingerprint)
    else:
        logger.info("Reusing cached image signatures for grouping.")

    signatures = {key: decode_signature(records[key]["signature"]) for key in keys}

    parent = {key: key for key in keys}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    def union_by(field: str) -> list[list[str]]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            value = records[key].get(field, "")
            if value and not value.startswith("error:"):
                buckets[value].append(key)
        clusters = [members for members in buckets.values() if len(members) > 1]
        for members in clusters:
            for extra in members[1:]:
                union(members[0], extra)
        return clusters

    byte_clusters = union_by("sha256")
    if byte_clusters:
        logger.warning(
            "Collapsed %d byte-identical duplicate clusters covering %d images.",
            len(byte_clusters),
            sum(len(m) for m in byte_clusters),
        )

    content_clusters = [
        cluster
        for cluster in union_by("content")
        if cluster not in byte_clusters
    ]
    if content_clusters:
        logger.warning(
            "Collapsed %d further clusters (%d images) whose pixels are identical "
            "at %dx%d despite differing bytes or filenames.",
            len(content_clusters),
            sum(len(m) for m in content_clusters),
            CONTENT_SIZE,
            CONTENT_SIZE,
        )

    class_of = dict(zip(keys, labels, strict=True)) if labels is not None else {}
    by_stem: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        by_stem[(class_of.get(key, ""), _numeric_stem(key))].append(key)

    verified_series = 0
    rejected_stems = 0
    independent_stems = 0
    for (cls, _stem), members in by_stem.items():
        if len(members) < 2:
            continue
        if cls in INDEPENDENT_CLASSES:
            # One file per subject by dataset design: never series-merge these.
            independent_stems += 1
            continue
        if _is_visual_series(members, signatures):
            verified_series += 1
            for extra in members[1:]:
                union(members[0], extra)
        else:
            rejected_stems += 1
            logger.debug(
                "Stem %r groups %d unrelated images; kept as independent samples.",
                _stem,
                len(members),
            )

    logger.info(
        "Grouping resolved %d images into %d source groups "
        "(%d verified series, %d shared-prefix stems rejected, "
        "%d independent-class stems kept unmerged).",
        len(keys),
        len({find(key) for key in keys}),
        verified_series,
        rejected_stems,
        independent_stems,
    )

    return [find(key) for key in keys]


def _is_visual_series(
    members: Sequence[str], signatures: dict[str, np.ndarray]
) -> bool:
    """True when consecutive frames of a named series are the same picture.

    Adjacent pairs are the meaningful comparison: in a slice series the first and
    last frames of a volume legitimately differ, so an all-pairs median would
    reject exactly the long series this grouping exists to catch.
    """
    ordered = sorted(members, key=_numeric_suffix)
    vectors = np.stack([signatures[key] for key in ordered])
    if vectors.shape[0] < 2 or not np.any(vectors):
        return False

    adjacent = np.sum(vectors[1:] * vectors[:-1], axis=1)
    return float(np.median(adjacent)) >= SERIES_CORRELATION_THRESHOLD


def _numeric_suffix(relative_path: str) -> tuple[int, str]:
    """Trailing integer of a filename, so frames sort in acquisition order."""
    stem = os.path.splitext(os.path.basename(relative_path))[0]
    trimmed = stem.rstrip("0123456789")
    tail = stem[len(trimmed) :]
    return (int(tail) if tail else 0, stem)


def _write_cache(
    cache_dir: str,
    keys: Sequence[str],
    records: dict[str, dict[str, str]],
    fingerprint: dict[str, Any],
) -> None:
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "records": {key: records[key] for key in keys},
    }
    written = False
    try:
        with open(_cache_path_for(cache_dir), "w") as handle:
            json.dump(payload, handle)
        written = True
    except OSError:
        pass

    if not written:
        fallback = os.path.join(tempfile.gettempdir(), ".case_group_cache.json")
        try:
            with open(fallback, "w") as handle:
                json.dump(payload, handle)
            logger.info("Persisted case-group cache to temporary directory: %s", fallback)
        except OSError as exc:
            logger.warning("Could not persist case-group cache: %s", exc)


def _grouped_stratified_assign(
    group_ids: Sequence[str],
    labels: Sequence[str],
    targets: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, keeping each class's proportions on target."""
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    group_sizes: Counter[str] = Counter()
    group_label: dict[str, str] = {}

    for group, label in zip(group_ids, labels, strict=True):
        group_sizes[group] += 1
        if group in group_label:
            continue
        group_label[group] = label
        groups_by_label[label].append(group)

    conflicts = report_label_conflicts(group_ids, labels)
    if conflicts:
        affected = sum(group_sizes[group] for group in conflicts)
        examples = "; ".join(
            f"{os.path.basename(group)} -> {sorted(names)}"
            for group, names in sorted(conflicts.items())[:3]
        )
        logger.warning(
            "%d group(s) covering %d sample(s) hold identical pixels with "
            "conflicting class labels. Grouping cannot fix contradictory "
            "supervision; repair DATA.json. Examples: %s",
            len(conflicts),
            affected,
            examples,
        )

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    filled: dict[str, Counter[str]] = {name: Counter() for name in targets}
    bucket_names = list(targets)

    for label in sorted(groups_by_label):
        groups = sorted(groups_by_label[label])
        order = rng.permutation(len(groups))
        total_label = sum(group_sizes[groups[i]] for i in range(len(groups)))

        for position in order:
            group = groups[position]
            size = group_sizes[group]
            best_bucket = max(
                bucket_names,
                key=lambda bucket: (
                    targets[bucket] * total_label - filled[bucket][label] - size / 2.0,
                    -size,
                    bucket,
                ),
            )
            assignment[group] = best_bucket
            filled[best_bucket][label] += size

    return assignment


def report_label_conflicts(
    group_ids: Sequence[str], labels: Sequence[str]
) -> dict[str, list[str]]:
    """Return groups whose members disagree on class, mapping group to its labels.

    Identical pixels given different labels is contradictory supervision: one of
    the two copies is mislabelled.
    """
    seen: dict[str, list[str]] = defaultdict(list)
    for group, label in zip(group_ids, labels, strict=True):
        if label not in seen[group]:
            seen[group].append(label)
    return {group: names for group, names in seen.items() if len(names) > 1}


def _report_split_quality(
    keys: Sequence[str],
    labels: Sequence[str],
    group_ids: Sequence[str],
    assignment: dict[str, str],
    buckets: dict[str, list[int]],
) -> None:
    """Verify isolation from the realised per-image assignment, not the intent."""
    split_of_index: dict[int, str] = {}
    for name, indices in buckets.items():
        for index in indices:
            split_of_index[index] = name

    missing = set(range(len(keys))) - set(split_of_index)
    if missing:
        raise RuntimeError(
            f"{len(missing)} sample(s) were not assigned to any split."
        )

    splits_per_group: dict[str, set[str]] = defaultdict(set)
    for index, group in enumerate(group_ids):
        splits_per_group[group].add(split_of_index[index])

    offending = {group: splits for group, splits in splits_per_group.items() if len(splits) > 1}
    leaked_samples = sum(group_ids.count(group) for group in offending)

    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for index, label in enumerate(labels):
        per_class[label][split_of_index[index]] += 1

    stragglers = [
        label
        for label, counts in per_class.items()
        if any(counts[bucket] == 0 for bucket in ("train", "val", "test"))
    ]

    logger.info(
        "Splits created: Train=%d, Val=%d, Test=%d | groups=%d | cross-split leaks=%d",
        len(buckets["train"]),
        len(buckets["val"]),
        len(buckets["test"]),
        len(splits_per_group),
        len(offending),
    )
    if stragglers:
        logger.warning(
            "%d class(es) are confined to a single split by their group structure "
            "and will not be represented everywhere: %s",
            len(stragglers),
            ", ".join(sorted(stragglers)[:5]),
        )
    if offending:
        examples = ", ".join(sorted(offending)[:3])
        raise RuntimeError(
            f"{len(offending)} source group(s) covering {leaked_samples} sample(s) "
            f"span multiple splits; refusing to train on leaked data. Examples: {examples}"
        )
