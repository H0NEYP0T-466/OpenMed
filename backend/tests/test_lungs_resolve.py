"""Unit tests for --resolve-conflicts cleaning pass in hygiene and trainer.

CPU-only, no GPU required, runs in <5s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.organs.lungs.classification.kaggle import hygiene
from app.organs.lungs.classification.kaggle.config import CLASS_NAMES, LungConfig
from app.organs.lungs.classification.kaggle.train_lungs_v2s import (
    parse_args,
    prepare_data,
)


def _build_synthetic_report(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Synthesize a tiny report.json in tmp_path with exact, conflict, tie and near clusters."""
    # 1 exact-dup group of 3 same-class paths
    exact_paths = [
        "COVID/images/ex1.png",
        "COVID/images/ex2.png",
        "COVID/images/ex3.png",
    ]
    # 1 cross-class conflict of 5 files split 3/2 across two classes
    c_covid = [
        "COVID/images/c1.png",
        "COVID/images/c2.png",
        "COVID/images/c3.png",
    ]
    c_normal = [
        "Normal/images/n1.png",
        "Normal/images/n2.png",
    ]
    # 1 cross-class conflict of 5 files split 2/2/1 (tie)
    tie_covid = [
        "COVID/images/t1.png",
        "COVID/images/t2.png",
    ]
    tie_normal = [
        "Normal/images/t3.png",
        "Normal/images/t4.png",
    ]
    tie_opacity = [
        "Lung_Opacity/images/t5.png",
    ]
    # 1 single-label near-dup cluster of 3
    near_viral = [
        "Viral Pneumonia/images/v1.png",
        "Viral Pneumonia/images/v2.png",
        "Viral Pneumonia/images/v3.png",
    ]

    report = {
        "generated_at": "2026-11-11T12:00:00+05:00",
        "dataset_root": str(tmp_path / "dataset"),
        "num_images": 16,
        "num_readable": 16,
        "num_unreadable": 0,
        "class_counts": {
            "COVID": 8,  # 3 exact + 3 c + 2 tie
            "Lung_Opacity": 1,  # 1 tie
            "Normal": 4,  # 2 c + 2 tie
            "Viral Pneumonia": 3,  # 3 near
        },
        "imbalance_ratio": 8.0,
        "exact_duplicates": {
            "groups": 1,
            "images": 3,
            "clusters": [
                {
                    "key": "sha256_mock_group_1",
                    "count": 3,
                    "classes": ["COVID"],
                    "paths": exact_paths,
                }
            ],
        },
        "cross_class_conflicts": [
            {
                "cluster_id": 101,
                "classes": ["COVID", "Normal"],
                "count": 5,
                "by_class": {
                    "COVID": c_covid,
                    "Normal": c_normal,
                },
            },
            {
                "cluster_id": 102,
                "classes": ["COVID", "Lung_Opacity", "Normal"],
                "count": 5,
                "by_class": {
                    "COVID": tie_covid,
                    "Normal": tie_normal,
                    "Lung_Opacity": tie_opacity,
                },
            },
        ],
        "near_duplicates": {
            "clusters": 1,
            "images": 3,
            "pairs": 3,
            "examples": [
                {
                    "cluster_id": 201,
                    "count": 3,
                    "paths": near_viral,
                }
            ],
        },
        "clusters": {"count": 4, "largest": 5},
    }

    report_path = tmp_path / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also create a stub report.md
    report_md_path = tmp_path / "report.md"
    report_md_path.write_text("# Lung dataset hygiene report\n\nInitial content.\n", encoding="utf-8")

    return report_path, report


def test_resolve_conflicts_counts_collapse_off(tmp_path: Path) -> None:
    report_path, _ = _build_synthetic_report(tmp_path)
    out_dir = tmp_path / "hygiene_out"

    manifest = hygiene.resolve_conflicts(
        report_json_path=report_path,
        output_dir=out_dir,
        collapse_single_label=False,
    )

    # 1. Exact dedup: 1 group of 3 -> keeps 1, drops 2
    assert manifest["dropped_exact"] == [
        "COVID/images/ex2.png",
        "COVID/images/ex3.png",
    ]

    # 2. Consensus relabel: 3 COVID vs 2 Normal -> relabels the 2 Normal to COVID
    assert manifest["relabeled"] == [
        {"path": "Normal/images/n1.png", "old_label": "Normal", "new_label": "COVID"},
        {"path": "Normal/images/n2.png", "old_label": "Normal", "new_label": "COVID"},
    ]

    # 2b. Tie cluster: 2 COVID, 2 Normal, 1 Lung_Opacity -> all 5 dropped
    assert sorted(manifest["dropped_tie"]) == [
        "COVID/images/t1.png",
        "COVID/images/t2.png",
        "Lung_Opacity/images/t5.png",
        "Normal/images/t3.png",
        "Normal/images/t4.png",
    ]

    # 3. Near dedup with collapse OFF -> keeps all 3, drops 0
    assert manifest["dropped_near_single"] == []

    # Kept count: 16 total - 2 exact - 5 tie - 0 near = 9
    assert manifest["kept_count"] == 9

    # Class counts recomputed
    expected_counts = {
        "COVID": 6,  # 8 - 2 dropped_exact + 2 relabeled - 2 tie
        "Lung_Opacity": 0,  # 1 - 1 tie
        "Normal": 0,  # 4 - 2 relabeled - 2 tie
        "Viral Pneumonia": 3,  # 3 - 0
    }
    assert manifest["class_counts_after"] == expected_counts
    assert sum(manifest["class_counts_after"].values()) == manifest["kept_count"]

    # Manifest file written
    assert (out_dir / "cleaned_manifest.json").is_file()

    # Report.md appended with ## Conflict resolution
    md_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Conflict resolution" in md_text
    assert "- Exact duplicates dropped: **2**" in md_text
    assert "- Cross-class conflicts relabeled (majority): **2**" in md_text
    assert "- Cross-class conflict ties dropped: **5**" in md_text
    assert "- Single-label near-duplicates collapsed: **0**" in md_text
    assert "- Kept images: **9**" in md_text


def test_resolve_conflicts_counts_collapse_on(tmp_path: Path) -> None:
    report_path, _ = _build_synthetic_report(tmp_path)
    out_dir = tmp_path / "hygiene_out_collapse"

    manifest = hygiene.resolve_conflicts(
        report_json_path=report_path,
        output_dir=out_dir,
        collapse_single_label=True,
    )

    # Near dedup with collapse ON -> keeps 1, drops 2
    assert manifest["dropped_near_single"] == [
        "Viral Pneumonia/images/v2.png",
        "Viral Pneumonia/images/v3.png",
    ]

    # Kept count: 16 - 2 exact - 5 tie - 2 near = 7
    assert manifest["kept_count"] == 7

    # Class counts recomputed
    expected_counts = {
        "COVID": 6,
        "Lung_Opacity": 0,
        "Normal": 0,
        "Viral Pneumonia": 1,  # 3 - 2 collapsed
    }
    assert manifest["class_counts_after"] == expected_counts
    assert sum(manifest["class_counts_after"].values()) == manifest["kept_count"]


def test_resolve_conflicts_apply_moves_files(tmp_path: Path) -> None:
    report_path, _ = _build_synthetic_report(tmp_path)
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "hygiene_apply"

    # Create dummy files
    for p in [
        "COVID/images/ex1.png",
        "COVID/images/ex2.png",
        "COVID/images/ex3.png",
        "COVID/images/c1.png",
        "Normal/images/n1.png",
        "COVID/images/t1.png",
        "Viral Pneumonia/images/v1.png",
    ]:
        fpath = dataset_dir / p
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b"dummy")

    manifest = hygiene.resolve_conflicts(
        report_json_path=report_path,
        output_dir=out_dir,
        apply=True,
        dataset_root=dataset_dir,
    )

    # ex2 was dropped exact -> moved to dropped/
    assert (out_dir / "dropped" / "COVID/images/ex2.png").is_file()
    assert not (dataset_dir / "COVID/images/ex2.png").is_file()

    # t1 was dropped tie -> moved to dropped/
    assert (out_dir / "dropped" / "COVID/images/t1.png").is_file()
    assert not (dataset_dir / "COVID/images/t1.png").is_file()

    # n1 was relabeled to COVID -> moved from Normal to COVID folder
    assert (dataset_dir / "COVID/images/n1.png").is_file()
    assert not (dataset_dir / "Normal/images/n1.png").is_file()


def test_cli_flags() -> None:
    # Default is False
    args_default = parse_args([])
    assert args_default.resolve_conflicts is False
    assert args_default.collapse_single_label is False
    assert args_default.apply_clean is False

    # Explicit flags
    args_flags = parse_args([
        "--resolve-conflicts",
        "--collapse-single-label",
        "--apply-clean",
    ])
    assert args_flags.resolve_conflicts is True
    assert args_flags.collapse_single_label is True
    assert args_flags.apply_clean is True


def test_prepare_data_warning_on_missing_report(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    cfg = LungConfig()
    dataset_dir = tmp_path / "dataset"
    seed = 0
    for cls in CLASS_NAMES:
        cls_dir = dataset_dir / cls / "images"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            seed += 1
            arr = np.random.RandomState(seed).randint(0, 256, (32, 32), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{i}.png")

    hygiene_dir = tmp_path / "hygiene_empty"
    hygiene_dir.mkdir(parents=True, exist_ok=True)

    # Should log warning and not crash when report.json does not exist
    records, cluster_members, plan, report = prepare_data(
        cfg=cfg,
        dataset_root=str(dataset_dir),
        hygiene_dir=hygiene_dir,
        resolve_conflicts=True,
    )
    assert len(records) == 16
    assert plan is not None
    assert (hygiene_dir / "report.json").is_file()


def test_prepare_data_with_resolve_conflicts_wired(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    cfg = LungConfig()
    dataset_dir = tmp_path / "dataset"
    hygiene_dir = tmp_path / "hygiene_run"
    hygiene_dir.mkdir(parents=True, exist_ok=True)

    # Build synthetic report in hygiene_dir
    report_path, report_data = _build_synthetic_report(hygiene_dir)

    for cls in CLASS_NAMES:
        cls_dir = dataset_dir / cls / "images"
        cls_dir.mkdir(parents=True, exist_ok=True)

    # Helper for distinct image
    seed_counter = 100

    def make_img(path: Path) -> None:
        nonlocal seed_counter
        seed_counter += 1
        arr = np.random.RandomState(seed_counter).randint(0, 256, (32, 32), dtype=np.uint8)
        Image.fromarray(arr).save(path)

    # 1. Exact paths (3 files in COVID, identical bytes)
    exact_arr = np.random.RandomState(42).randint(0, 256, (32, 32), dtype=np.uint8)
    for p in ["COVID/images/ex1.png", "COVID/images/ex2.png", "COVID/images/ex3.png"]:
        Image.fromarray(exact_arr).save(dataset_dir / p)

    # 2. Conflict majority paths (3 COVID, 2 Normal)
    for p in ["COVID/images/c1.png", "COVID/images/c2.png", "COVID/images/c3.png"]:
        make_img(dataset_dir / p)
    for p in ["Normal/images/n1.png", "Normal/images/n2.png"]:
        make_img(dataset_dir / p)

    # 3. Conflict tie paths (2 COVID, 2 Normal, 1 Lung_Opacity)
    for p in ["COVID/images/t1.png", "COVID/images/t2.png"]:
        make_img(dataset_dir / p)
    for p in ["Normal/images/t3.png", "Normal/images/t4.png"]:
        make_img(dataset_dir / p)
    make_img(dataset_dir / "Lung_Opacity/images/t5.png")

    # 4. Near dup paths (3 in Viral Pneumonia)
    for p in ["Viral Pneumonia/images/v1.png", "Viral Pneumonia/images/v2.png", "Viral Pneumonia/images/v3.png"]:
        make_img(dataset_dir / p)

    # Add 4 extra unique images per class so every class easily satisfies the >=3 cluster split requirement
    for cls in CLASS_NAMES:
        for k in range(4):
            make_img(dataset_dir / cls / "images" / f"extra_{k}.png")

    # Run prepare_data with resolve_conflicts=True
    records, cluster_members, plan, final_report = prepare_data(
        cfg=cfg,
        dataset_root=str(dataset_dir),
        hygiene_dir=hygiene_dir,
        resolve_conflicts=True,
        collapse_single_label=False,
    )

    # 16 synthetic + 16 extra = 32 images scanned
    # Dropped: 2 exact (ex2, ex3) + 5 tie (t1, t2, t3, t4, t5) = 7 dropped
    # Kept: 32 - 7 = 25 records!
    assert len(records) == 25

    rel_paths = {r.rel_path for r in records}
    assert "COVID/images/ex2.png" not in rel_paths
    assert "COVID/images/ex3.png" not in rel_paths
    assert "COVID/images/t1.png" not in rel_paths
    assert "Normal/images/t3.png" not in rel_paths
    assert "Lung_Opacity/images/t5.png" not in rel_paths

    # n1 and n2 were relabeled from Normal to COVID
    for r in records:
        if r.rel_path in ("Normal/images/n1.png", "Normal/images/n2.png"):
            assert r.class_name == "COVID"
            assert r.class_index == CLASS_NAMES.index("COVID")

    # Check cleaned_manifest.json exists
    assert (hygiene_dir / "cleaned_manifest.json").is_file()
    # Check report.md has ## Conflict resolution
    md_text = (hygiene_dir / "report.md").read_text(encoding="utf-8")
    assert "## Conflict resolution" in md_text

