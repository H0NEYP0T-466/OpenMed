"""Brain tumour classification dataset, manifest discovery and leakage-free splitting.

Splitting is group-aware: images verified to originate from the same source
scan are forced into a single split, and byte-identical duplicates are
collapsed into the same group regardless of filename. See
`app/organs/brain/README.md` for the measurement behind the grouping rule.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SIGNATURE_SIZE = 32
SERIES_CORRELATION_THRESHOLD = 0.90
CACHE_FORMAT_VERSION = 1


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


def _signature(path: str) -> list[float]:
    """Mean-removed, unit-normalised low-resolution luminance signature."""
    with Image.open(path) as image:
        gray = image.convert("L").resize(
            (SIGNATURE_SIZE, SIGNATURE_SIZE), Image.Resampling.BILINEAR
        )
    array = np.asarray(gray, dtype=np.float32)
    array -= array.mean()
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array /= norm
    return array.reshape(-1).tolist()


def _probe_image(args: tuple[str, str]) -> tuple[str, str, list[float]]:
    key, absolute = args
    try:
        return key, _sha256(absolute), _signature(absolute)
    except Exception as exc:  # noqa: BLE001 - recorded and surfaced by caller
        logger.warning("Unreadable image during grouping probe: %s (%s)", key, exc)
        return key, f"error:{exc}", []


class BrainTumorDataset(Dataset):
    """Slice-level brain MRI classification dataset driven by a DATA.json manifest."""

    MASK_SUFFIX = "_mask.png"

    def __init__(
        self,
        data_root: str,
        transform: Any | None = None,
        split_indices: Sequence[int] | None = None,
    ) -> None:
        self.json_path, self.images_base = self.locate_data_and_images(data_root)
        self.data_root = os.path.dirname(self.json_path)
        self.transform = transform

        with open(self.json_path) as handle:
            raw_data = json.load(handle)

        self.class_names: list[str] = sorted(
            {
                metadata["class"]
                for path, metadata in raw_data.items()
                if not path.endswith(self.MASK_SUFFIX)
            }
        )
        self.tumor_types: list[str] = sorted(
            {
                metadata["tumor_type"]
                for path, metadata in raw_data.items()
                if not path.endswith(self.MASK_SUFFIX)
            }
        )
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        records = [
            (path, metadata)
            for path, metadata in raw_data.items()
            if not path.endswith(self.MASK_SUFFIX)
        ]
        records.sort(key=lambda item: item[0])

        self.samples: list[dict[str, Any]] = [
            {
                "path": os.path.join(self.images_base, path),
                "relative_path": path,
                "metadata": metadata,
            }
            for path, metadata in records
        ]

        if split_indices is not None:
            self.samples = [self.samples[i] for i in split_indices]

        logger.info(
            "Loaded dataset with %d samples. (%d classes)",
            len(self.samples),
            len(self.class_names),
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
            if self.transform is not None:
                image = self.transform(image)

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
    ) -> torch.Tensor:
        """Square-root smoothed inverse-frequency weights, mean-normalised and clipped."""
        counts = Counter(sample["metadata"]["class"] for sample in dataset_samples)
        num_classes = len(class_names)
        total = len(dataset_samples) or 1

        weights = np.ones(num_classes, dtype=np.float32)
        for idx, name in enumerate(class_names):
            count = counts.get(name, 0)
            if count > 0:
                weights[idx] = np.sqrt(total / (num_classes * count))

        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights /= mean_weight

        return torch.tensor(np.clip(weights, 0.2, 5.0), dtype=torch.float)

    @staticmethod
    def get_stratified_splits(
        data_root: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return leak-free train/val/test indices grouped by source scan."""
        json_path, images_base = BrainTumorDataset.locate_data_and_images(data_root)
        with open(json_path) as handle:
            raw_data = json.load(handle)

        keys = sorted(
            path for path in raw_data if not path.endswith(BrainTumorDataset.MASK_SUFFIX)
        )
        labels = [raw_data[key]["class"] for key in keys]

        group_ids = build_case_groups(
            json_path=json_path,
            images_base=images_base,
            keys=keys,
            seed=seed,
        )

        targets = {
            "train": max(0.0, 1.0 - test_size - val_size),
            "val": val_size,
            "test": test_size,
        }
        assignment = _grouped_stratified_assign(group_ids, labels, targets, seed)

        buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for index, key in enumerate(keys):
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


def _cache_path_for(json_path: str) -> str:
    return os.path.join(os.path.dirname(json_path), ".case_group_cache.json")


def _load_cache(json_path: str, keys: Sequence[str]) -> dict[str, dict[str, str]] | None:
    path = _cache_path_for(json_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable case-group cache: %s", exc)
        return None

    if payload.get("format") != CACHE_FORMAT_VERSION:
        return None
    if payload.get("manifest_mtime") != os.path.getmtime(json_path):
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

    for key, digest, signature in probed:
        results[key] = {
            "sha256": digest,
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
    json_path: str,
    images_base: str,
    keys: Sequence[str],
    seed: int = 42,
    workers: int | None = None,
) -> list[str]:
    """Map every image to a group id such that one group never spans two splits.

    Two images share a group when they are byte-identical, or when they share a
    numeric filename stem *and* are measurably the same picture. The correlation
    test is what separates true slice series from unrelated scans that merely
    share a naming prefix.
    """
    if workers is None:
        workers = min(8, os.cpu_count() or 1)

    records = _load_cache(json_path, keys)
    if records is None:
        logger.info("Computing image signatures for leakage-safe grouping...")
        records = _probe_images(images_base, keys, workers)
        _write_cache(json_path, keys, records)
    else:
        logger.info("Reusing cached image signatures for grouping.")

    digests = {key: records[key]["sha256"] for key in keys}
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

    by_digest: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        digest = digests[key]
        if digest and not digest.startswith("error:"):
            by_digest[digest].append(key)

    duplicate_groups = [members for members in by_digest.values() if len(members) > 1]
    for members in duplicate_groups:
        for extra in members[1:]:
            union(members[0], extra)
    if duplicate_groups:
        logger.warning(
            "Collapsed %d byte-identical duplicate clusters covering %d images.",
            len(duplicate_groups),
            sum(len(m) for m in duplicate_groups),
        )

    by_stem: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_stem[_numeric_stem(key)].append(key)

    verified_series = 0
    rejected_stems = 0
    for stem, members in by_stem.items():
        if len(members) < 2:
            continue
        if _is_visual_series(members, signatures):
            verified_series += 1
            for extra in members[1:]:
                union(members[0], extra)
        else:
            rejected_stems += 1
            logger.debug(
                "Stem %r groups %d unrelated images; kept as independent samples.",
                stem,
                len(members),
            )

    logger.info(
        "Grouping resolved %d images into %d source groups "
        "(%d verified series, %d shared-prefix stems rejected).",
        len(keys),
        len({find(key) for key in keys}),
        verified_series,
        rejected_stems,
    )

    return [find(key) for key in keys]


def _is_visual_series(
    members: Sequence[str], signatures: dict[str, np.ndarray]
) -> bool:
    matrix = np.stack([signatures[key] for key in members])
    if matrix.shape[0] < 2 or not np.any(matrix):
        return False

    grams = matrix @ matrix.T
    take_upper = np.triu_indices(matrix.shape[0], k=1)
    pairwise = grams[take_upper]
    if pairwise.size == 0:
        return False
    return float(np.median(pairwise)) >= SERIES_CORRELATION_THRESHOLD


def _write_cache(
    json_path: str, keys: Sequence[str], records: dict[str, dict[str, str]]
) -> None:
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "manifest_mtime": os.path.getmtime(json_path),
        "count": len(keys),
        "records": {key: records[key] for key in keys},
    }
    try:
        with open(_cache_path_for(json_path), "w") as handle:
            json.dump(payload, handle)
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

    for group, label in zip(group_ids, labels):
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
    for group, label in zip(group_ids, labels):
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
    seen: dict[str, str] = {}
    crossings = 0
    for group, index in zip(group_ids, range(len(keys))):
        split = assignment[group]
        if group in seen and seen[group] != split:
            crossings += 1
        seen[group] = split

    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for index, label in enumerate(labels):
        per_class[label][assignment[group_ids[index]]] += 1

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
        len(set(group_ids)),
        crossings,
    )
    if stragglers:
        logger.warning(
            "%d class(es) are confined to a single split by their group structure "
            "and will not be represented everywhere: %s",
            len(stragglers),
            ", ".join(sorted(stragglers)[:5]),
        )
    if crossings:
        raise RuntimeError(
            "Grouped split failed isolation invariant; refusing to train on leaked data."
        )
