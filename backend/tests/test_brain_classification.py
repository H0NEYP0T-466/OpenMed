"""Regression tests for the brain classification module.

The split-isolation tests are the important ones: a group that appears in two
splits silently inflates reported accuracy, which is the failure this module
previously shipped with.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.organs.brain.classification import brain_regions
from app.organs.brain.classification.dataset import BrainTumorDataset
from app.organs.brain.classification.label_space import (
    CLASS_NAMES,
    LabelSpaceError,
    label_space_path_for,
    resolve_label_space,
    save_label_space,
    split_class_name,
)

REAL_ARCHIVE = os.getenv(
    "OPENMED_BRAIN_DATASET",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "brain", "archive"),
)


def _write_image(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.uint8(array)).convert("L").convert("RGB").save(path, "JPEG")


def _smooth_field(rng: np.random.Generator, size: int = 64, blocks: int = 6) -> np.ndarray:
    """Low-frequency field, so small shifts stay highly correlated like real slices."""
    coarse = rng.integers(0, 255, size=(blocks, blocks), dtype=np.uint8)
    image = Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _series_frame(base: np.ndarray, index: int) -> np.ndarray:
    """Translate the pattern slightly per frame: correlated, but not identical."""
    return np.roll(np.roll(base, index, axis=0), index // 2, axis=1)


def _build_manifest_dataset(
    root: str,
    classes: dict[str, int],
    series_base: np.ndarray,
    unrelated_base: np.ndarray,
) -> dict[str, Any]:
    """Create DATA.json plus images: one true slice series and one name-prefix trap."""
    images_root = os.path.join(root, "Images_")
    records: dict[str, Any] = {}

    def add(relative: str, label: str, array: np.ndarray) -> None:
        absolute = os.path.join(images_root, relative)
        _write_image(absolute, array)
        tumor, _, sequence = label.rpartition(" ")
        records[relative] = {
            "filename": os.path.basename(relative),
            "class": label,
            "tumor_type": tumor,
            "sequence": sequence,
            "width": 64,
            "height": 64,
            "point": {},
            "location": [],
            "has_lesion": 1 if tumor != "Normal" else 0,
            "bbox_path": "",
            "mask_path": "",
        }

    # A genuine series: near-identical frames sharing a numeric filename stem.
    for index in range(classes["series"]):
        add(
            f"Meningioma/Meningioma T1/Meningioma T1/T1 - Meningioma frontal {index:03d}.jpg",
            "Meningioma T1",
            _series_frame(series_base, index),
        )

    # Look-alike names that are visually unrelated (catalogue indices, not slices).
    for index in range(classes["unrelated"]):
        rng = np.random.default_rng(1000 + index)
        add(
            f"Normal/Normal T1/Normal T1/normal_sweep_{index:03d}.jpg",
            "Normal T1",
            rng.integers(0, 255, size=(64, 64), dtype=np.uint8),
        )

    # A third class, spread across two clearly distinct source groups.
    for group in range(classes["groups"]):
        add(
            f"Glioma/Glioma T2/Glioma T2/T2 - Glioblastoma temporal case{group}.jpg",
            "Glioblastoma T2",
            np.clip(unrelated_base.astype(np.int16) + group * 40, 0, 255).astype(np.uint8),
        )

    with open(os.path.join(root, "DATA.json"), "w") as handle:
        json.dump(records, handle)
    return records


@pytest.fixture()
def synthetic_dataset(tmp_path):
    root = str(tmp_path / "ds")
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(7)
    records = _build_manifest_dataset(
        root,
        {"series": 12, "unrelated": 12, "groups": 8},
        _smooth_field(rng),
        rng.integers(0, 255, size=(64, 64), dtype=np.uint8),
    )
    return root, records


def _splits(data_root: str) -> tuple[list[int], list[int], list[int]]:
    return BrainTumorDataset.get_stratified_splits(data_root, test_size=0.2, val_size=0.2, seed=7)


def _split_map(data_root: str, records: dict[str, Any]) -> dict[str, str]:
    train, val, test = _splits(data_root)
    keys = sorted(records)
    mapping: dict[str, str] = {}
    for name, indices in (("train", train), ("val", val), ("test", test)):
        for index in indices:
            mapping[keys[index]] = name
    return mapping


def test_no_group_crosses_splits(synthetic_dataset) -> None:
    root, records = synthetic_dataset
    train, val, test = _splits(root)
    keys = sorted(records)
    json_path, images_base = BrainTumorDataset.locate_data_and_images(root)
    groups = BrainTumorDataset.build_group_ids(json_path, images_base, keys)

    seen: dict[str, set[str]] = defaultdict(set)
    train_set, val_set, test_set = set(train), set(val), set(test)
    for index, group in enumerate(groups):
        if index in train_set:
            seen[group].add("train")
        elif index in val_set:
            seen[group].add("val")
        elif index in test_set:
            seen[group].add("test")

    crossings = {group: splits for group, splits in seen.items() if len(splits) > 1}
    assert crossings == {}, f"groups leaked across splits: {crossings}"


def test_slices_of_one_series_stay_together(synthetic_dataset) -> None:
    root, records = synthetic_dataset
    mapping = _split_map(root, records)
    series_splits = {
        mapping[key] for key in records if "Meningioma frontal" in key
    }
    assert len(series_splits) == 1, "a 12-frame series was split apart"


def test_shared_name_prefix_does_not_imply_same_source(synthetic_dataset) -> None:
    root, records = synthetic_dataset
    json_path, images_base = BrainTumorDataset.locate_data_and_images(root)
    keys = sorted(k for k in records if "normal_sweep" in k)
    groups = BrainTumorDataset.build_group_ids(json_path, images_base, keys)

    # `normal_sweep_000..012` share a stem but are unrelated pictures, so they
    # must not be collapsed into one group.
    assert len(set(groups)) == len(keys)


def test_byte_identical_files_are_grouped_despite_different_names(tmp_path) -> None:
    root = str(tmp_path / "dupes")
    images = os.path.join(root, "Images_")
    payload = np.random.default_rng(3).integers(0, 255, size=(64, 64), dtype=np.uint8)

    first = "Class A/Class A T1/Class A T1/T1 - Case alpha 004.jpg"
    second = "Class B/Class B T1C+/Class B T1C+/T1C+ - Case beta 020.jpg"
    for relative in (first, second):
        _write_image(os.path.join(images, relative), payload)

    records = {
        relative: {
            "filename": os.path.basename(relative),
            "class": "Astrocytoma T1" if relative == first else "Astrocytoma T1C+",
            "tumor_type": "Astrocytoma",
            "sequence": "T1" if relative == first else "T1C+",
            "point": {},
            "location": [],
            "has_lesion": 1,
        }
        for relative in (first, second)
    }
    with open(os.path.join(root, "DATA.json"), "w") as handle:
        json.dump(records, handle)

    keys = sorted(records)
    groups = BrainTumorDataset.build_group_ids(
        os.path.join(root, "DATA.json"), images, keys
    )
    assert groups[0] == groups[1], "identical bytes were treated as independent scans"


def test_classes_with_several_sources_spread_over_all_splits(synthetic_dataset) -> None:
    """A class is only testable everywhere if it has more than one source group.

    Grouped splitting makes this trade-off explicit: a class represented by a
    single scan must live entirely in one split, which is honest rather than
    leaking that scan into the others.
    """
    root, records = synthetic_dataset
    mapping = _split_map(root, records)
    json_path, images_base = BrainTumorDataset.locate_data_and_images(root)
    keys = sorted(records)
    groups = BrainTumorDataset.build_group_ids(json_path, images_base, keys)

    groups_per_class: dict[str, set[str]] = defaultdict(set)
    splits_per_class: dict[str, set[str]] = defaultdict(set)
    for key, group in zip(keys, groups):
        label = records[key]["class"]
        groups_per_class[label].add(group)
        splits_per_class[label].add(mapping[key])

    assert set(splits_per_class) == {"Meningioma T1", "Normal T1", "Glioblastoma T2"}
    for label, group_ids in groups_per_class.items():
        if len(group_ids) >= 3:
            assert len(splits_per_class[label]) == 3, f"{label} should appear in all splits"

    # The 12-frame series is one source, so it must be confined to one split.
    assert len(groups_per_class["Meningioma T1"]) == 1
    assert len(splits_per_class["Meningioma T1"]) == 1


def test_splits_are_disjoint_and_complete(synthetic_dataset) -> None:
    root, records = synthetic_dataset
    train, val, test = _splits(root)
    total = len(records)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == total


def test_dataset_exposes_labels_per_instance(synthetic_dataset) -> None:
    root, _ = synthetic_dataset
    dataset = BrainTumorDataset(root, transform=None)
    assert dataset.class_names == sorted(dataset.class_names)
    assert "Meningioma T1" in dataset.class_names
    # The previous implementation cached class names on the class object, which
    # leaked one dataset's label space into the next.
    assert not hasattr(BrainTumorDataset, "CLASS_NAMES")


def test_class_weights_are_bounded_and_mean_one(synthetic_dataset) -> None:
    root, _ = synthetic_dataset
    dataset = BrainTumorDataset(root, transform=None)
    weights = BrainTumorDataset.compute_class_weights(dataset.samples, dataset.class_names)
    assert weights.numel() == len(dataset.class_names)
    assert float(weights.min()) >= 0.2
    assert float(weights.max()) <= 5.0


def test_corrupted_sample_yields_a_real_image_not_a_black_square(synthetic_dataset) -> None:
    """The old implementation fed an all-black image plus a genuine label into training."""
    root, _ = synthetic_dataset
    dataset = BrainTumorDataset(root, transform=None)
    victim_path = dataset.samples[0]["path"]
    substitute = dataset.samples[1]["metadata"]["class"]

    with open(victim_path, "wb") as handle:
        handle.write(b"deliberately not an image")

    image, label, meta = dataset[0]

    assert isinstance(image, Image.Image)
    assert meta["relative_path"] != dataset.samples[0]["relative_path"]
    assert label == dataset.class_to_idx[substitute]
    assert np.asarray(image).std() > 1.0, "returned a uniform blank image"


def test_isolation_guard_actually_fires(tmp_path) -> None:
    """The guard must reject a crossing assignment rather than trust the intent."""
    from app.organs.brain.classification import dataset as dataset_module

    group_ids = ["a", "a", "b"]
    labels = ["Normal T1", "Normal T1", "Normal T2"]
    buckets = {"train": [0], "val": [1], "test": [2]}

    with pytest.raises(RuntimeError, match="span multiple splits"):
        dataset_module._report_split_quality(
            ["k0", "k1", "k2"], labels, group_ids, {}, buckets
        )

    with pytest.raises(RuntimeError, match="not assigned"):
        dataset_module._report_split_quality(
            ["k0", "k1", "k2"], labels, group_ids, {}, {"train": [0], "val": [1], "test": []}
        )


def test_label_space_refuses_width_mismatch() -> None:
    with pytest.raises(LabelSpaceError, match="does not match"):
        resolve_label_space(None, 39)


def test_label_space_accepts_matching_width() -> None:
    space = resolve_label_space(None, len(CLASS_NAMES))
    assert space.num_classes == len(CLASS_NAMES)
    assert space.class_names == tuple(CLASS_NAMES)


def test_label_space_sidecar_is_preferred_and_validated(tmp_path) -> None:
    checkpoint = str(tmp_path / "model.pth")
    open(checkpoint, "wb").close()

    names = [*CLASS_NAMES, "Extra T1"]
    save_label_space(checkpoint, names, model_tag="tag", metrics={"test_acc": 0.5})
    space = resolve_label_space(checkpoint, len(names))
    assert space.source.startswith("sidecar:")
    assert space.metrics["test_acc"] == 0.5

    with pytest.raises(LabelSpaceError, match="has a 39-class head"):
        resolve_label_space(checkpoint, 39)

    assert label_space_path_for(checkpoint).endswith("model.label_space.json")


@pytest.mark.parametrize(
    "class_name, expected",
    [
        ("Ependymoma - Subependymoma T1C+", ("Ependymoma - Subependymoma", "T1C+")),
        ("Normal T2", ("Normal", "T2")),
        ("Dysembryoplastic Neuroepithelial Tumor T1", ("Dysembryoplastic Neuroepithelial Tumor", "T1")),
        ("Mystery", ("Mystery", "Unknown")),
    ],
)
def test_split_class_name(class_name: str, expected: tuple[str, str]) -> None:
    assert split_class_name(class_name) == expected


def test_regions_report_millimetres_and_scale_with_confidence() -> None:
    high = brain_regions.map_prediction_to_3d("Meningioma T1", None, 0.9)
    low = brain_regions.map_prediction_to_3d("Meningioma T1", None, 0.2)

    assert high and low
    assert any(abs(value) > 5 for value in high[0]["coordinates_3d"]), "coordinates are not in mm"
    assert high[0]["probability"] > low[0]["probability"]
    assert high[0]["basis"] == "tumour_type_prior"


def test_normal_maps_to_no_region() -> None:
    assert brain_regions.map_prediction_to_3d("Normal T1", None, 0.98) == []


def test_every_registry_entry_resolves_to_a_centroid() -> None:
    for name, region in brain_regions.BRAIN_REGIONS_3D.items():
        assert region["name"] == name
        assert len(region["coordinates_3d"]) == 3
        assert brain_regions.resolve_region(name) is region


def test_alias_table_targets_exist() -> None:
    for alias, target in brain_regions.LOCATION_ALIASES.items():
        assert target in brain_regions.BRAIN_REGIONS_3D, f"alias {alias!r} points nowhere"


@pytest.fixture(scope="module")
def tiny_backbone():
    pytest.importorskip("torch")
    from app.organs.brain.classification.model import create_model, resolve_gradcam_layer

    model = create_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.eval()
    return model, resolve_gradcam_layer(model)


def test_gradcam_reuse_produces_fresh_maps(tiny_backbone) -> None:
    """Hooks were removed after the first call, so later samples replayed sample one."""
    model, layer = tiny_backbone
    torch = pytest.importorskip("torch")
    from app.organs.brain.classification.model import GradCAM

    first = np.random.default_rng(11).random((1, 3, 208, 208)).astype("float32")
    second = np.random.default_rng(12).random((1, 3, 208, 208)).astype("float32")

    with GradCAM(model, layer) as grad_cam:
        map_a = grad_cam.generate(torch.from_numpy(first), 0)
        map_b = grad_cam.generate(torch.from_numpy(second), 5)

    assert map_a.shape == map_b.shape
    assert not np.allclose(map_a, map_b), "second map reused tensors from the first"
    assert grad_cam._handles == [], "context exit should detach hooks"


def test_gradcam_rejects_batched_input(tiny_backbone) -> None:
    model, layer = tiny_backbone
    torch = pytest.importorskip("torch")
    from app.organs.brain.classification.model import GradCAM

    with GradCAM(model, layer) as grad_cam:
        with pytest.raises(ValueError, match="exactly one image"):
            grad_cam.generate(torch.zeros(2, 3, 208, 208))


def test_overlay_preserves_geometry() -> None:
    from app.organs.brain.classification.model import overlay_cam_on_image

    image = np.random.default_rng(5).integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
    blended = overlay_cam_on_image(image, np.random.default_rng(6).random((7, 7)))
    assert blended.shape == image.shape
    assert blended.dtype == np.uint8


def test_pipeline_refuses_random_weights(tmp_path) -> None:
    from app.organs.brain.classification.pipeline import (
        BrainClassificationPipeline,
        ModelUnavailableError,
    )

    with pytest.raises(ModelUnavailableError, match="random weights"):
        BrainClassificationPipeline(model_path=None)

    with pytest.raises(ModelUnavailableError, match="not found"):
        BrainClassificationPipeline(model_path=str(tmp_path / "absent.pth"))


def test_pipeline_rejects_legacy_head_width(tmp_path) -> None:
    """A 39-class checkpoint against 42 names must fail closed, not relabel."""
    torch = pytest.importorskip("torch")
    from app.organs.brain.classification.model import create_model
    from app.organs.brain.classification.pipeline import (
        BrainClassificationPipeline,
        ModelUnavailableError,
    )

    checkpoint = tmp_path / "legacy.pth"
    torch.save(create_model(num_classes=39, pretrained=False).state_dict(), checkpoint)

    with pytest.raises(ModelUnavailableError, match="does not match"):
        BrainClassificationPipeline(model_path=str(checkpoint))


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(REAL_ARCHIVE, "DATA.json")),
    reason="brain dataset not present",
)
def test_all_real_dataset_locations_resolve() -> None:
    with open(os.path.join(REAL_ARCHIVE, "DATA.json")) as handle:
        raw = json.load(handle)
    tags = {
        location
        for metadata in raw.values()
        for location in (metadata.get("location") or [])
    }
    unresolved = sorted(tag for tag in tags if brain_regions.resolve_region(tag) is None)
    assert unresolved == []


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(REAL_ARCHIVE, "DATA.json")),
    reason="brain dataset not present",
)
def test_grouped_split_is_isolated_on_the_real_dataset() -> None:
    json_path, images_base = BrainTumorDataset.locate_data_and_images(REAL_ARCHIVE)
    with open(json_path) as handle:
        raw = json.load(handle)
    keys = sorted(k for k in raw if not k.endswith(BrainTumorDataset.MASK_SUFFIX))
    groups = BrainTumorDataset.build_group_ids(json_path, images_base, keys)
    train, val, test = BrainTumorDataset.get_stratified_splits(REAL_ARCHIVE)

    buckets = [set(train), set(val), set(test)]
    per_group: dict[str, set[int]] = defaultdict(set)
    for position, group in enumerate(groups):
        for bucket_index, bucket in enumerate(buckets):
            if position in bucket:
                per_group[group].add(bucket_index)

    assert all(len(buckets) == 1 for buckets in per_group.values())
    assert len(train) + len(val) + len(test) == len(keys)
