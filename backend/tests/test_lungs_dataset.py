"""Dataset, transform and group-split tests for the lung training package.

Three contracts are pinned here because each one, if broken, produces a number
that looks fine and is not:

* class order — index *i* of the head must be ``CLASS_NAMES[i]`` everywhere;
* transform output — 3 x 300 x 300 float32, grayscale replicated, ImageNet
  normalised, at both training and eval resolution;
* the split — a duplicate cluster never straddles two splits.

Plus the Kaggle path resolver, which is the only thing standing between the
trainer and "silently trained on one class".
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.organs.lungs.classification.kaggle import dataset as lung_dataset
from app.organs.lungs.classification.kaggle import hygiene
from app.organs.lungs.classification.kaggle.config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    NUM_CLASSES,
    LungConfig,
    canonical_class,
    class_dirs,
    list_class_images,
    resolve_dataset_root,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _png(path: Path, seed: int, size: int = 96) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.random.default_rng(seed).integers(0, 255, size=(size, size), dtype=np.uint8)
    Image.fromarray(array).convert("L").save(path)
    return path


def _record(path: Path, class_index: int, cluster_id: int) -> hygiene.ImageRecord:
    return hygiene.ImageRecord(
        path=str(path),
        rel_path=str(path.name),
        class_name=CLASS_NAMES[class_index],
        class_index=class_index,
        sha256=f"sha-{cluster_id}-{path.name}",
        phash=cluster_id,
        width=96,
        height=96,
        mode="L",
        cluster_id=cluster_id,
        source="test",
    )


def _synthetic_manifest(seed: int, clusters_per_class: int = 10):
    """Clusters of 1-3 images, each cluster single-class, all classes covered."""
    rng = random.Random(seed)
    records: list[hygiene.ImageRecord] = []
    clusters: dict[int, list[int]] = {}
    cluster_id = 0
    for class_index in range(NUM_CLASSES):
        for _ in range(clusters_per_class):
            size = rng.choice([1, 1, 1, 2, 3])
            members: list[int] = []
            for member in range(size):
                index = len(records)
                records.append(
                    _record(
                        Path(f"/synthetic/c{cluster_id}_{member}.png"),
                        class_index,
                        cluster_id,
                    )
                )
                members.append(index)
            clusters[cluster_id] = members
            cluster_id += 1
    return records, clusters


# ─────────────────────────────────────────────────────────────────────────────
# Class order
# ─────────────────────────────────────────────────────────────────────────────


def test_class_order_is_exactly_the_contract_order() -> None:
    assert CLASS_NAMES == ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")
    assert NUM_CLASSES == 4
    assert CLASS_TO_INDEX == {
        "COVID": 0,
        "Lung_Opacity": 1,
        "Normal": 2,
        "Viral Pneumonia": 3,
    }


def test_serving_module_declares_the_same_order() -> None:
    from app.organs.lungs.classification import model_v2s

    assert model_v2s.CLASS_NAMES == CLASS_NAMES
    assert model_v2s.NUM_CLASSES == NUM_CLASSES


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("COVID", "COVID"),
        ("covid19", "COVID"),
        ("Lung_Opacity", "Lung_Opacity"),
        ("Lung Opacity", "Lung_Opacity"),
        ("LUNG-OPACITY", "Lung_Opacity"),
        ("Normal", "Normal"),
        ("Viral Pneumonia", "Viral Pneumonia"),
        ("Viral_Pneumonia", "Viral Pneumonia"),
        ("ViralPneumonia", "Viral Pneumonia"),
        ("masks", None),
    ],
)
def test_class_folder_names_are_matched_tolerantly(folder: str, expected) -> None:
    assert canonical_class(folder) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle path resolution
# ─────────────────────────────────────────────────────────────────────────────


def _make_mixed_layout(root: Path) -> None:
    """Half the classes use ``<class>/images/*.png``, half are flattened."""
    for index, class_name in enumerate(CLASS_NAMES):
        folder = root / ("Lung Opacity" if class_name == "Lung_Opacity" else class_name)
        if index % 2 == 0:
            target = folder / "images"
            target.mkdir(parents=True, exist_ok=True)
            _png(target / f"{index}_0.png", index * 10 + 1)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            _png(folder / f"{index}_0.png", index * 10 + 1)


def test_resolver_finds_a_nested_kaggle_input_layout(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle" / "input"
    slug = input_root / "datasets" / "owner" / "covid19-radiography-database"
    _make_mixed_layout(slug)

    resolved = Path(resolve_dataset_root(str(input_root)))

    assert resolved == slug
    assert resolved.is_dir()


def test_resolver_accepts_the_flat_mount_layout(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle" / "input"
    flat = input_root / "covid19-radiography-database" / "COVID-19_Radiography_Dataset"
    _make_mixed_layout(flat)

    assert Path(resolve_dataset_root(str(input_root))) == flat


def test_resolver_prefers_the_shallowest_complete_directory(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    outer = input_root / "datasets" / "owner" / "slug"
    _make_mixed_layout(outer)
    # A decoy that is deeper, incomplete, and must lose.
    decoy = outer / "extra" / "deeper"
    (decoy / "COVID").mkdir(parents=True)

    assert Path(resolve_dataset_root(str(input_root))) == outer


def test_resolver_reports_what_it_saw_when_a_class_is_missing(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    root = input_root / "slug"
    _make_mixed_layout(root)
    shutil.rmtree(root / "Normal")

    with pytest.raises(FileNotFoundError, match="Could not locate"):
        resolve_dataset_root(str(input_root))


def test_resolver_rejects_a_missing_input_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_dataset_root(str(tmp_path / "nope"))


def test_class_dirs_maps_every_canonical_class(tmp_path: Path) -> None:
    _make_mixed_layout(tmp_path)

    mapped = class_dirs(tmp_path)

    assert list(mapped) == list(CLASS_NAMES)
    assert mapped["Lung_Opacity"].name == "Lung Opacity"
    # Even-indexed classes keep the ``images`` subfolder, odd ones are flat.
    assert (mapped["COVID"] / "images").is_dir()
    assert not (mapped["Lung_Opacity"] / "images").is_dir()


def test_list_class_images_handles_both_layouts(tmp_path: Path) -> None:
    nested = tmp_path / "COVID"
    flat = tmp_path / "Normal"
    for index in range(2):
        _png(nested / "images" / f"{index}.png", index)
        _png(flat / f"{index}.png", index)
    (nested / "images" / "notes.txt").write_text("ignored")

    nested_images = list_class_images(nested)
    flat_images = list_class_images(flat)

    assert len(nested_images) == 2
    assert len(flat_images) == 2
    assert all(path.suffix == ".png" for path in nested_images + flat_images)
    assert [path.name for path in nested_images] == ["0.png", "1.png"]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    return _png(tmp_path / "scan.png", 3, size=256)


def test_eval_transform_returns_3x300x300_float(image_file: Path) -> None:
    transform = lung_dataset.build_eval_transform(300)
    tensor = transform(lung_dataset.load_rgb(image_file))

    assert tuple(tensor.shape) == (3, 300, 300)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_train_transform_returns_3x300x300_float(image_file: Path) -> None:
    transform = lung_dataset.build_train_transform(300, LungConfig())
    tensor = transform(lung_dataset.load_rgb(image_file))

    assert tuple(tensor.shape) == (3, 300, 300)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_eval_transform_is_imagenet_normalised_grayscale(image_file: Path) -> None:
    tensor = lung_dataset.build_eval_transform(300)(lung_dataset.load_rgb(image_file))

    # ImageNet normalisation pushes 0 and 1 well outside [-1, 1].
    assert float(tensor.min()) < -1.5
    assert float(tensor.max()) > 1.5
    assert float(tensor.min()) >= -2.2
    assert float(tensor.max()) <= 2.7

    # Denormalised, the three channels are the same luminance (the model must
    # not learn a colour cue from a greyscale modality). Compared before the
    # uint8 round-trip, which can differ by one level between channels.
    mean = torch.tensor(lung_dataset.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(lung_dataset.IMAGENET_STD).view(3, 1, 1)
    pixels = tensor * std + mean
    assert torch.allclose(pixels[0], pixels[1], atol=1e-6)
    assert torch.allclose(pixels[0], pixels[2], atol=1e-6)
    assert float(pixels.min()) >= -1e-6
    assert float(pixels.max()) <= 1.0 + 1e-6


def test_eval_transform_works_at_384(image_file: Path) -> None:
    tensor = lung_dataset.build_eval_transform(384)(lung_dataset.load_rgb(image_file))
    assert tuple(tensor.shape) == (3, 384, 384)


def test_load_rgb_flattens_colour_to_luminance(tmp_path: Path) -> None:
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[:, :, 0] = 255  # pure red: luminance is not 255
    source = tmp_path / "red.png"
    Image.fromarray(array).save(source)

    image = lung_dataset.load_rgb(source)
    channels = np.asarray(image)

    assert image.mode == "RGB"
    assert np.array_equal(channels[:, :, 0], channels[:, :, 1])
    assert int(channels[0, 0, 0]) < 255


# ─────────────────────────────────────────────────────────────────────────────
# LungDataset
# ─────────────────────────────────────────────────────────────────────────────


def test_dataset_item_returns_tensor_label_and_record_index(image_file: Path) -> None:
    record = _record(image_file, class_index=2, cluster_id=0)
    dataset = lung_dataset.LungDataset(
        [record], [0], lung_dataset.build_eval_transform(300)
    )

    tensor, label, record_index = dataset[0]

    assert tuple(tensor.shape) == (3, 300, 300)
    assert label == 2
    assert record_index == 0
    assert len(dataset) == 1


def test_minority_class_uses_the_stronger_transform(image_file: Path) -> None:
    """The minority boost must change the *distribution*, not redraw it.

    Two independent draws from one transform keep that transform's distribution,
    so a second draw is statistically a no-op. The boost is only real if the
    minority class is routed to a distinct, stronger policy.
    """

    class CountingTransform:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            return torch.full((3, 300, 300), float(len(self.tag)))

    records = [
        _record(image_file, class_index=3, cluster_id=0),  # Viral Pneumonia
        _record(image_file, class_index=0, cluster_id=1),  # COVID
    ]
    base = CountingTransform("base")
    strong = CountingTransform("stronger")
    dataset = lung_dataset.LungDataset(
        records,
        [0, 1],
        base,
        minority_class_index=3,
        minority_extra_draw_prob=1.0,
        minority_transform=strong,
    )

    minority_tensor, _, _ = dataset[0]
    assert strong.calls == 1, "the minority class must route to the stronger policy"
    assert base.calls == 0, "the minority sample must not also be drawn by the base policy"
    assert float(minority_tensor[0, 0, 0]) == float(len("stronger"))

    _, _, _ = dataset[1]
    assert base.calls == 1, "non-minority classes draw exactly once"
    assert strong.calls == 1


def test_minority_boost_without_a_stronger_policy_is_a_no_op(image_file: Path) -> None:
    """With no minority_transform supplied the sampler must not waste a draw."""

    class CountingTransform:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            return torch.zeros(3, 300, 300)

    records = [_record(image_file, class_index=3, cluster_id=0)]
    transform = CountingTransform()
    dataset = lung_dataset.LungDataset(
        records,
        [0],
        transform,
        minority_class_index=3,
        minority_extra_draw_prob=1.0,
    )

    dataset[0]
    assert transform.calls == 1


def test_minority_boost_is_off_by_default(image_file: Path) -> None:
    class CountingTransform:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            return torch.zeros(3, 300, 300)

    records = [_record(image_file, class_index=3, cluster_id=0)]
    transform = CountingTransform()
    dataset = lung_dataset.LungDataset(records, [0], transform)

    dataset[0]
    assert transform.calls == 1


# ─────────────────────────────────────────────────────────────────────────────
# Group-aware split
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_cluster_ever_spans_two_splits(seed: int) -> None:
    records, clusters = _synthetic_manifest(seed)
    plan = lung_dataset.grouped_stratified_split(
        records, clusters, CLASS_NAMES, seed=seed
    )

    split_of_cluster: dict[int, set[str]] = {}
    for cluster_id, members in clusters.items():
        assigned = {plan.split_of[index] for index in members}
        split_of_cluster[cluster_id] = assigned
        assert len(assigned) == 1, f"cluster {cluster_id} landed in {assigned}"

    # The guard agrees, and every image is assigned exactly once.
    lung_dataset.assert_cluster_isolation(records, clusters, plan.split_of)
    assert len(plan.split_of) == len(records)
    assert sum(plan.counts().values()) == len(records)
    assert not (set(plan.train) & set(plan.val))
    assert not (set(plan.train) & set(plan.test))
    assert not (set(plan.val) & set(plan.test))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_is_stratified_and_roughly_70_15_15(seed: int) -> None:
    records, clusters = _synthetic_manifest(seed)
    plan = lung_dataset.grouped_stratified_split(
        records, clusters, CLASS_NAMES, seed=seed
    )

    total = len(records)
    counts = plan.counts()
    assert 0.60 <= counts["train"] / total <= 0.80
    assert 0.08 <= counts["val"] / total <= 0.22
    assert 0.08 <= counts["test"] / total <= 0.22

    for class_name in CLASS_NAMES:
        per_split = plan.per_class[class_name]
        assert all(count > 0 for count in per_split.values()), (
            f"{class_name} missing from a split: {per_split}"
        )


def test_isolation_guard_actually_fires() -> None:
    records, clusters = _synthetic_manifest(0)
    torn = dict.fromkeys(range(len(records)), "train")
    torn[clusters[0][0]] = "test"

    with pytest.raises(RuntimeError, match="span more than one split"):
        lung_dataset.assert_cluster_isolation(records, clusters, torn)


def test_split_refuses_a_class_with_too_few_clusters() -> None:
    records = [
        _record(Path("/synthetic/a.png"), 0, 0),
        _record(Path("/synthetic/b.png"), 0, 1),
        _record(Path("/synthetic/c.png"), 1, 2),
        _record(Path("/synthetic/d.png"), 1, 3),
        _record(Path("/synthetic/e.png"), 2, 4),
        _record(Path("/synthetic/f.png"), 2, 5),
        _record(Path("/synthetic/g.png"), 3, 6),
        _record(Path("/synthetic/h.png"), 3, 7),
    ]
    clusters = {index: [index] for index in range(len(records))}

    with pytest.raises(RuntimeError, match="only 2 duplicate cluster"):
        lung_dataset.grouped_stratified_split(records, clusters, CLASS_NAMES)


def test_split_rejects_impossible_fractions() -> None:
    records, clusters = _synthetic_manifest(0)
    with pytest.raises(ValueError, match="must leave a training split"):
        lung_dataset.grouped_stratified_split(
            records, clusters, CLASS_NAMES, val_frac=0.6, test_frac=0.5
        )


def test_class_weights_are_inverse_frequency_with_mean_one() -> None:
    records = [
        _record(Path("/synthetic/a.png"), 0, 0),
        _record(Path("/synthetic/b.png"), 0, 1),
        _record(Path("/synthetic/c.png"), 0, 2),
        _record(Path("/synthetic/d.png"), 0, 3),
        _record(Path("/synthetic/e.png"), 1, 4),
        _record(Path("/synthetic/f.png"), 1, 5),
        _record(Path("/synthetic/g.png"), 2, 6),
        _record(Path("/synthetic/h.png"), 3, 7),
    ]
    weights = lung_dataset.compute_class_weights(list(range(len(records))), records, 4)

    assert weights.shape == (4,)
    assert float(weights.mean()) == pytest.approx(1.0)
    # Inverse frequency: 4, 2, 1, 1 before normalisation.
    assert weights[0] < weights[1] < weights[2]
    assert weights[2] == pytest.approx(weights[3])


def test_class_weights_refuse_a_class_absent_from_the_split() -> None:
    records = [
        _record(Path("/synthetic/a.png"), 0, 0),
        _record(Path("/synthetic/b.png"), 0, 1),
        _record(Path("/synthetic/c.png"), 1, 2),
        _record(Path("/synthetic/d.png"), 1, 3),
        _record(Path("/synthetic/e.png"), 2, 4),
        _record(Path("/synthetic/f.png"), 2, 5),
        _record(Path("/synthetic/g.png"), 3, 6),
    ]
    with pytest.raises(RuntimeError, match="absent from the training split"):
        lung_dataset.compute_class_weights([0, 2, 4], records, 4)


def test_split_manifest_covers_every_split_once() -> None:
    records, clusters = _synthetic_manifest(0)
    plan = lung_dataset.grouped_stratified_split(records, clusters, CLASS_NAMES, seed=0)

    rows = lung_dataset.manifest_rows(records, plan)

    assert len(rows) == len(records)
    assert {row[5] for row in rows} == {"train", "val", "test"}
    assert all(row[0] and row[1] and row[2] for row in rows)
