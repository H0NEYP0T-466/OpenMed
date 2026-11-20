"""Cluster purity, patient grouping and the grouped split.

A split built on images rather than on clusters leaks. This module's whole reason
for existing is that the shipped COVID-QU-Ex split leaks: two pixel-identical
pairs cross the boundary, and the COVID-19 class carries BIDS-style names whose
``sub-`` prefix identifies a *patient*, so one person's sessions can land in both
train and test.

So the assertions are:

* a duplicate cluster is never split across two splits, whatever else happens;
* two files carrying the same subject id are always in the same cluster, even
  when their pixels are different — that is the patient leak, and it is the one
  the original brief did not cover;
* the split keeps the three classes and fills all three splits.
"""

from __future__ import annotations

from collections import Counter

import pytest

from app.organs.lungs.segmentation.kaggle.catalog import Record
from app.organs.lungs.segmentation.kaggle.clusters import build_clusters
from app.organs.lungs.segmentation.kaggle.config import SPLITS, TrainConfig
from app.organs.lungs.segmentation.kaggle.splits import (
    assert_cluster_purity,
    build_split,
    cross_split_counts,
    load_manifest,
    write_manifest,
)
from app.organs.lungs.segmentation.labels import CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# purity: the one invariant that must never bend
# ─────────────────────────────────────────────────────────────────────────────


def test_purity_accepts_a_clean_partition() -> None:
    cluster_of = {0: 0, 1: 1, 2: 1, 3: 2}
    split_of = {0: "train", 1: "val", 2: "val", 3: "test"}
    assert_cluster_purity(cluster_of, split_of)   # cluster 1 sits wholly in val


def test_purity_rejects_a_cluster_that_spans_two_splits() -> None:
    cluster_of = {0: 0, 1: 1, 2: 1}
    split_of = {0: "train", 1: "val", 2: "test"}
    with pytest.raises(AssertionError, match="cluster"):
        assert_cluster_purity(cluster_of, split_of)


# ─────────────────────────────────────────────────────────────────────────────
# clustering
# ─────────────────────────────────────────────────────────────────────────────


def _clusters(subject_keys: dict[int, str] | None = None, *, n: int = 8):
    uids = list(range(n))
    sha_bytes = {uid: f"bytes-{uid}" for uid in uids}
    sha_pixels = {uid: f"pixels-{uid}" for uid in uids}
    # far-apart hashes so the perceptual pass can only ever confirm what is planted
    phashes = {uid: (uid + 1) * 0x0101010101010101 for uid in uids}
    paths = {uid: f"/nope/{uid}.png" for uid in uids}
    return build_clusters(
        uids, sha_bytes, sha_pixels, phashes, paths,
        phash_hamming_max=0, pixel_mae=0.0, min_corr=0.99, bucket_cap=8,
        subject_keys=subject_keys,
    )


def test_identical_pixels_are_one_cluster() -> None:
    uids = [0, 1, 2, 3]
    clusters = build_clusters(
        uids,
        {0: "a", 1: "b", 2: "c", 3: "d"},
        {0: "same", 1: "same", 2: "x", 3: "y"},    # 0 and 1 decode to the same pixels
        {uid: (uid + 1) * 0x0101010101010101 for uid in uids},
        {uid: f"/nope/{uid}.png" for uid in uids},
        phash_hamming_max=0, pixel_mae=0.0, min_corr=0.99, bucket_cap=8,
    )
    assert clusters.cluster_of[0] == clusters.cluster_of[1]
    assert clusters.cluster_of[2] != clusters.cluster_of[0]


def test_same_subject_is_one_cluster_even_with_different_pixels() -> None:
    """The patient leak: two sessions, same ``sub-S…`` id, different images."""
    clusters = _clusters({2: "S09760", 3: "S09760"})

    assert clusters.cluster_of[2] == clusters.cluster_of[3]
    assert clusters.stats["subject_ids"] == 1
    assert clusters.stats["subject_unions"] == 1
    assert clusters.stats["largest_subject_group"] == 2


def test_subject_grouping_is_off_when_no_keys_are_given() -> None:
    clusters = _clusters(None)
    assert clusters.stats["subject_ids"] == 0
    assert len({clusters.cluster_of[uid] for uid in range(8)}) == 8


def test_unrelated_images_stay_separate() -> None:
    clusters = _clusters()
    assert len({clusters.cluster_of[uid] for uid in range(8)}) == 8


# ─────────────────────────────────────────────────────────────────────────────
# the split itself
# ─────────────────────────────────────────────────────────────────────────────


def _records(n: int = 60) -> list[Record]:
    classes = (CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL)
    records = []
    for uid in range(n):
        cls = classes[uid % 3]
        shipped = ("train", "val", "test")[uid % 3]          # catalog keys are lowercase
        records.append(Record(
            uid=uid, filename=f"{cls.lower()}_{uid}.png", cls=cls, shipped_split=shipped,
            image_path=f"/nope/{uid}.png", lung_mask_path=f"/nope/{uid}_lung.png",
            in_infection_tree=cls != CLASS_NON_COVID and uid % 2 == 0,
            infection_mask_path=(f"/nope/{uid}_inf.png"
                                 if cls != CLASS_NON_COVID and uid % 2 == 0 else None),
        ))
    return records


def _split(records: list[Record]):
    uids = [r.uid for r in records]
    clusters = build_clusters(
        uids,
        {uid: f"bytes-{uid}" for uid in uids},
        {uid: f"pixels-{uid}" for uid in uids},          # all distinct: no duplicates
        {uid: (uid + 1) * 0x0101010101010101 for uid in uids},
        {uid: f"/nope/{uid}.png" for uid in uids},
        phash_hamming_max=0, pixel_mae=0.0, min_corr=0.99, bucket_cap=16,
        subject_keys={2: "S0002", 3: "S0002"},           # one planted patient
    )
    return build_split(records, records, clusters, uids, TrainConfig())


def test_split_fills_three_splits_and_keeps_every_image() -> None:
    records = _records()
    result = _split(records)

    for name in SPLITS:
        assert result.part(name), f"{name} is empty"
    assert sum(len(result.part(name)) for name in SPLITS) == len(records)


def test_split_purity_holds_on_the_real_builder() -> None:
    result = _split(_records())
    # raises if any cluster straddles two splits
    assert_cluster_purity(result.cluster_of, result.split_of)


def test_split_keeps_classes_in_every_split() -> None:
    result = _split(_records())
    for name in SPLITS:
        present = Counter(record.cls for record in result.part(name))
        assert set(present) == {CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL}


def test_split_report_states_the_cross_split_counts() -> None:
    result = _split(_records())
    report = result.report
    for key in ("cross_split_before", "cross_split_after", "counts_after", "moved_matrix",
                "images_moved", "purity_asserted"):
        assert key in report
    assert report["purity_asserted"] is True
    assert report["cross_split_after"]["clusters_spanning_splits"] == 0


def test_manifest_round_trips_through_disk(tmp_path) -> None:
    """A resumed run rebuilds the partition from the CSV alone; it must be lossless."""
    result = _split(_records())
    path = str(tmp_path / "split_manifest.csv")
    write_manifest(result, path)

    reloaded = load_manifest(path)
    for name in SPLITS:
        original = sorted(record.uid for record in result.part(name))
        restored = sorted(record.uid for record in reloaded.part(name))
        assert original == restored
    assert reloaded.cluster_of == result.cluster_of


def test_cross_split_counts_finds_the_leak() -> None:
    """The shipped split does leak; the counter has to see it."""
    records = _records(12)
    clusters = build_clusters(
        [r.uid for r in records],
        {r.uid: "same-bytes" for r in records},           # everything is one duplicate
        {r.uid: "same-pixels" for r in records},
        {r.uid: (r.uid + 1) * 0x0101010101010101 for r in records},
        {r.uid: f"/nope/{r.uid}.png" for r in records},
        phash_hamming_max=0, pixel_mae=0.0, min_corr=0.99, bucket_cap=16,
    )
    shipped = {r.uid: r.shipped_split for r in records}
    counts = cross_split_counts(clusters, set(shipped), shipped)
    assert counts["clusters_spanning_splits"] >= 1
    assert counts["exact_byte_pairs"] > 0
