"""Hygiene / leakage-audit tests for the lung dataset.

The audit is the reason the split can be trusted at all: it is what turns
"byte-identical files under two labels" from an invisible macro-F1 inflation
into a listed conflict and a cluster that the splitter refuses to break apart.

Every fixture here is a handful of 256x256 synthetic images written into
``tmp_path``. No dataset, no network, no GPU.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.organs.lungs.classification.kaggle import hygiene
from app.organs.lungs.classification.kaggle.config import CLASS_NAMES, class_dirs

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic images
# ─────────────────────────────────────────────────────────────────────────────


def _smooth(seed: int, size: int = 256, blocks: int = 6) -> Image.Image:
    """Low-frequency field: stable under resize, so pHash survives re-encoding."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(20, 235, size=(blocks, blocks), dtype=np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.BICUBIC)


def _write_classes(root: Path, per_class: int = 3, seed_base: int = 100) -> None:
    """``per_class`` visually distinct images per class, in the Kaggle layout."""
    for class_index, class_name in enumerate(CLASS_NAMES):
        images = root / class_name / "images"
        images.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            _smooth(seed_base + class_index * 10 + index).save(images / f"{index}.png")


def _near_duplicate(source: Path, target: Path) -> None:
    """Resize + JPEG re-encode: same picture, different bytes."""
    with Image.open(source) as image:
        buffer = io.BytesIO()
        image.resize((200, 200), Image.BICUBIC).save(buffer, "JPEG", quality=85)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        decoded.convert("L").save(target, "PNG")


def _corrupt(path: Path) -> None:
    path.write_bytes(b"this is not a PNG, whatever the extension says")


@pytest.fixture()
def dirty_dataset(tmp_path: Path) -> Path:
    """Four classes with one of each defect the audit must catch.

    * exact duplicate: ``COVID/images/2.png`` is a byte copy of
      ``Normal/images/1.png``  → exact dupe *and* cross-class conflict
    * near duplicate: ``Lung_Opacity/images/1.png`` is a resized/recompressed
      ``Lung_Opacity/images/0.png``
    * corrupt: ``Viral Pneumonia/images/2.png``
    """
    root = tmp_path / "dataset"
    _write_classes(root)

    shutil.copyfile(
        root / "Normal" / "images" / "1.png", root / "COVID" / "images" / "2.png"
    )
    _near_duplicate(
        root / "Lung_Opacity" / "images" / "0.png",
        root / "Lung_Opacity" / "images" / "1.png",
    )
    _corrupt(root / "Viral Pneumonia" / "images" / "2.png")
    return root


@pytest.fixture()
def clean_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "clean"
    _write_classes(root)
    return root


def _audit(root: Path, hamming_max: int = 5, split_of=None):
    """Scan → cluster → report, exactly as ``--audit-only`` does."""
    records = hygiene.scan(class_dirs(root), CLASS_NAMES, root, workers=2)
    hygiene.assign_sources(records, root)
    clusters, stats = hygiene.cluster_records(records, hamming_max)
    report = hygiene.build_report(records, root, clusters, hamming_max, split_of, stats)
    return records, clusters, report


def _by_rel(records, rel_path: str):
    for record in records:
        if record.rel_path.replace(os.sep, "/") == rel_path:
            return record
    raise AssertionError(f"{rel_path} not scanned")


# ─────────────────────────────────────────────────────────────────────────────
# Hashing primitives
# ─────────────────────────────────────────────────────────────────────────────


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"openmed-lungs-hygiene"
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert hygiene.sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_hamming_counts_differing_bits() -> None:
    assert hygiene.hamming(0b1010, 0b1010) == 0
    assert hygiene.hamming(0b1010, 0b0000) == 2
    assert hygiene.hamming(0, (1 << hygiene.PHASH_BITS) - 1) == hygiene.PHASH_BITS


def test_phash_of_the_same_picture_survives_resize(tmp_path: Path) -> None:
    original = tmp_path / "a.png"
    resized = tmp_path / "b.png"
    _smooth(7).save(original)
    _near_duplicate(original, resized)
    assert hygiene.hamming(hygiene.phash_file(original), hygiene.phash_file(resized)) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# The audit on the dirty fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_scan_reports_exact_counts(dirty_dataset: Path) -> None:
    _, _, report = _audit(dirty_dataset)

    assert report["num_images"] == 12
    assert report["num_readable"] == 11
    assert report["num_unreadable"] == 1
    assert report["class_counts"] == {
        "COVID": 3,
        "Lung_Opacity": 3,
        "Normal": 3,
        "Viral Pneumonia": 2,
    }
    assert report["class_counts_ordered"] == [3, 3, 3, 2]
    assert report["imbalance_ratio"] == pytest.approx(1.5)


def test_exact_duplicate_pair_is_one_group_of_two(dirty_dataset: Path) -> None:
    records, _, report = _audit(dirty_dataset)

    covid = _by_rel(records, "COVID/images/2.png")
    normal = _by_rel(records, "Normal/images/1.png")
    assert covid.sha256 == normal.sha256

    exact = report["exact_duplicates"]
    assert exact["groups"] == 1
    assert exact["images"] == 2
    group = exact["clusters"][0]
    assert group["count"] == 2
    assert sorted(group["paths"]) == ["COVID/images/2.png", "Normal/images/1.png"]
    assert group["classes"] == ["COVID", "Normal"]


def test_near_duplicate_pair_is_grouped_without_being_byte_identical(
    dirty_dataset: Path,
) -> None:
    records, _, report = _audit(dirty_dataset)

    first = _by_rel(records, "Lung_Opacity/images/0.png")
    second = _by_rel(records, "Lung_Opacity/images/1.png")
    assert first.sha256 != second.sha256
    assert hygiene.hamming(first.phash, second.phash) <= 5

    near = report["near_duplicates"]
    assert near["clusters"] == 1
    assert near["images"] == 2
    assert near["pairs"] == 1
    assert sorted(near["examples"][0]["paths"]) == [
        "Lung_Opacity/images/0.png",
        "Lung_Opacity/images/1.png",
    ]


def test_cluster_ids_are_exact_and_shared_by_duplicates(dirty_dataset: Path) -> None:
    records, clusters, report = _audit(dirty_dataset)

    assert report["clusters"]["count"] == 9
    assert sorted(clusters) == list(range(9))
    assert report["clusters"]["largest"] == 2
    assert report["clusters"]["histogram"] == {"1": 7, "2": 2}

    covid = _by_rel(records, "COVID/images/2.png")
    normal = _by_rel(records, "Normal/images/1.png")
    opacity_first = _by_rel(records, "Lung_Opacity/images/0.png")
    opacity_second = _by_rel(records, "Lung_Opacity/images/1.png")

    # The cross-class exact pair is cluster 2; the near-duplicate pair is
    # cluster 3. Both are the only multi-member clusters in the fixture.
    assert covid.cluster_id == 2
    assert normal.cluster_id == 2
    assert opacity_first.cluster_id == 3
    assert opacity_second.cluster_id == 3
    assert sorted(clusters[2]) == sorted([records.index(covid), records.index(normal)])
    assert sorted(clusters[3]) == sorted(
        [records.index(opacity_first), records.index(opacity_second)]
    )

    # Every other readable image is alone in its cluster.
    singleton_ids = [
        record.cluster_id for record in records if record.readable and record.cluster_id not in (2, 3)
    ]
    assert len(singleton_ids) == 7
    assert len(set(singleton_ids)) == 7
    # Unreadable files never enter a cluster.
    assert _by_rel(records, "Viral Pneumonia/images/2.png").cluster_id == -1


def test_cross_class_conflict_is_listed_not_dropped(dirty_dataset: Path) -> None:
    _, _, report = _audit(dirty_dataset)

    conflicts = report["cross_class_conflicts"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["cluster_id"] == 2
    assert conflict["classes"] == ["COVID", "Normal"]
    assert conflict["count"] == 2
    assert conflict["by_class"]["COVID"] == ["COVID/images/2.png"]
    assert conflict["by_class"]["Normal"] == ["Normal/images/1.png"]
    # Both files survive into the report's readable count — nothing is dropped.
    assert report["num_readable"] == 11


def test_unreadable_file_is_reported_with_its_error(dirty_dataset: Path) -> None:
    _, _, report = _audit(dirty_dataset)

    assert len(report["unreadable"]) == 1
    entry = report["unreadable"][0]
    assert entry["path"] == "Viral Pneumonia/images/2.png"
    assert entry["error"]


def test_source_pools_come_from_the_class_folder(dirty_dataset: Path) -> None:
    records, _, report = _audit(dirty_dataset)

    pools = {record.class_name: record.source for record in records if record.readable}
    assert pools == {
        "COVID": "covid_mixed",
        "Lung_Opacity": "rsna",
        "Normal": "rsna",
        "Viral Pneumonia": "viral",
    }
    assert report["leakage"]["computed"] is False
    assert report["leakage"]["clusters_spanning_splits"] is None


def test_dirty_dataset_verdict_is_fail(dirty_dataset: Path) -> None:
    _, _, report = _audit(dirty_dataset)

    assert report["verdict"] == "FAIL"
    assert any("unreadable" in reason for reason in report["verdict_reasons"])
    assert any("two different labels" in reason for reason in report["verdict_reasons"])


# ─────────────────────────────────────────────────────────────────────────────
# Verdict transitions
# ─────────────────────────────────────────────────────────────────────────────


def test_verdict_flips_from_pass_to_fail(clean_dataset: Path) -> None:
    _, _, report = _audit(clean_dataset)

    assert report["num_images"] == 12
    assert report["num_unreadable"] == 0
    assert report["exact_duplicates"]["groups"] == 0
    assert report["near_duplicates"]["clusters"] == 0
    assert report["cross_class_conflicts"] == []
    assert report["verdict"] == "PASS", report["verdict_reasons"]

    # One corrupt file out of 13 is above the 2% tolerance → FAIL, not WARN.
    _corrupt(clean_dataset / "Normal" / "images" / "2.png")
    _, _, after = _audit(clean_dataset)

    assert after["num_images"] == 12  # the file is still counted, just unreadable
    assert after["num_unreadable"] == 1
    assert after["verdict"] == "FAIL"
    assert after["verdict"] != report["verdict"]


def test_duplicates_alone_are_warn_not_fail(dirty_dataset: Path) -> None:
    """With the corrupt file removed the only defects are duplicates → WARN."""
    (dirty_dataset / "Viral Pneumonia" / "images" / "2.png").unlink()
    _, _, report = _audit(dirty_dataset)

    assert report["num_unreadable"] == 0
    assert report["exact_duplicates"]["groups"] == 1
    assert report["verdict"] == "WARN"
    assert any("two different labels" in reason for reason in report["verdict_reasons"])


# ─────────────────────────────────────────────────────────────────────────────
# Leakage audit
# ─────────────────────────────────────────────────────────────────────────────


def test_cluster_spanning_splits_fails_the_verdict(dirty_dataset: Path) -> None:
    records = hygiene.scan(class_dirs(dirty_dataset), CLASS_NAMES, dirty_dataset, workers=2)
    hygiene.assign_sources(records, dirty_dataset)
    clusters, stats = hygiene.cluster_records(records, 5)

    readable = [index for index, record in enumerate(records) if record.readable]
    split_of: dict[int, str] = {}
    for position, cluster_id in enumerate(sorted(clusters)):
        # Every cluster whole, alternating by cluster (never by image) ...
        target = ("train", "val", "test")[position % 3]
        for member in clusters[cluster_id]:
            split_of[member] = target
    # ... except the cross-class cluster, deliberately torn in half.
    split_of[clusters[2][0]] = "train"
    split_of[clusters[2][1]] = "test"
    assert len(split_of) == len(readable)

    report = hygiene.build_report(records, dirty_dataset, clusters, 5, split_of, stats)
    leakage = report["leakage"]

    assert leakage["computed"] is True
    assert leakage["clusters_spanning_splits"] == 1
    assert leakage["examples"][0]["cluster_id"] == 2
    assert leakage["examples"][0]["splits"] == ["test", "train"]
    assert report["verdict"] == "FAIL"
    assert any("span two splits" in reason for reason in report["verdict_reasons"])


def test_clean_split_audits_as_zero_leakage(clean_dataset: Path) -> None:
    records = hygiene.scan(class_dirs(clean_dataset), CLASS_NAMES, clean_dataset, workers=2)
    hygiene.assign_sources(records, clean_dataset)
    clusters, stats = hygiene.cluster_records(records, 5)

    split_of = {
        index: ("train" if index < 6 else "val" if index < 9 else "test")
        for index, record in enumerate(records)
        if record.readable
    }
    report = hygiene.build_report(records, clean_dataset, clusters, 5, split_of, stats)

    assert report["leakage"]["clusters_spanning_splits"] == 0
    assert report["leakage"]["split_counts"] == {"train": 6, "val": 3, "test": 3}
    assert report["leakage"]["source_pools_in_multiple_splits"] >= 1
    assert report["verdict"] == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# Artefacts
# ─────────────────────────────────────────────────────────────────────────────


def test_run_audit_writes_json_csv_and_markdown(dirty_dataset: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "hygiene"
    records, clusters, report = hygiene.run_audit(
        class_dirs(dirty_dataset), CLASS_NAMES, dirty_dataset, out_dir, workers=2
    )

    assert len(records) == 12
    assert report["verdict"] == "FAIL"

    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    csv_path = out_dir / "report.csv"
    assert json_path.is_file() and md_path.is_file() and csv_path.is_file()

    with open(json_path, encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["verdict"] == "FAIL"
    assert written["num_images"] == 12
    assert written["clusters"]["count"] == len(clusters)

    markdown = md_path.read_text(encoding="utf-8")
    assert "Verdict: `FAIL`" in markdown
    assert "Cross-class conflicts" in markdown

    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == [
        "rel_path",
        "class",
        "sha256",
        "phash",
        "cluster_id",
        "width",
        "height",
        "mode",
        "readable",
        "source",
        "split",
    ]
