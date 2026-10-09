"""Tests for the MedNeXt segmentation package.

Covers the architecture (shape, deep supervision, export/strict reload, UpKern),
the train/serve preprocessing contract, augmentation invariants, losses and
metrics, hygiene decisions, split construction (with a negative control for the
leakage check), checkpoint validation, the serving pipeline, and a full CPU
smoke run of ``train_mednext.py`` on synthetic data.

No GPU and no real dataset are needed.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
import torch

from app.organs.brain.segmentation.mednext import model as M
from app.organs.brain.segmentation.mednext import pipeline as P
from app.organs.brain.segmentation.mednext import preprocessor as PRE
from app.organs.brain.segmentation.mednext.training import audit as A
from app.organs.brain.segmentation.mednext.training import augment as AUG
from app.organs.brain.segmentation.mednext.training import config as C
from app.organs.brain.segmentation.mednext.training import export as X
from app.organs.brain.segmentation.mednext.training import losses as L
from app.organs.brain.segmentation.mednext.training import metrics as MET
from app.organs.brain.segmentation.mednext.training import splits as S

# ── synthetic data ────────────────────────────────────────────────────────


def _head(size: int, rng: np.random.Generator, drift: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = size / 2 + drift, size / 2 - drift
    inside = ((yy - cy) / (size * 0.42)) ** 2 + ((xx - cx) / (size * 0.34)) ** 2 < 1.0
    texture = cv2.GaussianBlur(rng.normal(90, 40, (size, size)).astype(np.float32), (0, 0), 3)
    return np.where(inside, np.clip(texture, 30, 200), 0.0).astype(np.float32)


def write_pair(image_path: str, mask_path: str, size: int, seed: int, colour: bool) -> None:
    rng = np.random.default_rng(seed)
    head = _head(size, rng, drift=rng.uniform(-4, 4))
    yy, xx = np.mgrid[0:size, 0:size]
    cy = int(size * rng.uniform(0.4, 0.6))
    cx = int(size * rng.uniform(0.4, 0.6))
    radius = size * rng.uniform(0.06, 0.12)
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius**2
    image = np.where(blob, np.clip(head + 80, 0, 255), head).astype(np.uint8)
    mask = (blob & (head > 0)).astype(np.uint8) * 255
    if not mask.any():
        mask[cy - 2:cy + 2, cx - 2:cx + 2] = 255
        image[cy - 2:cy + 2, cx - 2:cx + 2] = 220
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if colour else image)
    cv2.imwrite(mask_path, mask)


def make_sources(root, n_openmed: int = 120, n_btsc: int = 120, size: int = 128) -> tuple[str, str]:
    om = os.path.join(str(root), "openmed")
    bt = os.path.join(str(root), "btsc")
    for base in (om, bt):
        os.makedirs(os.path.join(base, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, "masks"), exist_ok=True)
    kinds = ["Glioma frontal", "Meningioma parietal", "Schwannoma cerebellar", "Pituitary sellar",
             "Ependymoma ventricle", "Astrocytoma temporal", "Medulloblastoma vermis",
             "Craniopharyngioma sellar", "Germinoma pineal", "Hemangioblastoma occipital"]
    per = n_openmed // len(kinds)
    for k, kind in enumerate(kinds):
        for j in range(per):
            stem = f"T1 - {kind} {j + 1:03d}"
            write_pair(os.path.join(om, "images", stem + ".jpg"),
                       os.path.join(om, "masks", stem + ".png"), size, 1000 * k + j, colour=True)
    for i in range(1, n_btsc + 1):
        write_pair(os.path.join(bt, "images", f"{i}.png"), os.path.join(bt, "masks", f"{i}.png"),
                   size, 50_000 + i, colour=False)
    return om, bt


@pytest.fixture(scope="session")
def sources(tmp_path_factory):
    return make_sources(tmp_path_factory.mktemp("synthetic"))


def tiny_spec(**kw) -> M.MedNeXtSpec:
    return M.MedNeXtSpec.from_variant("S", n_channels=8, **kw)


# ── architecture ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("variant,expected_m", [("S", 5.47), ("B", 10.45), ("M", 17.42), ("L", 61.56)])
def test_variant_parameter_counts_match_paper(variant, expected_m):
    model = M.build_model(M.MedNeXtSpec.from_variant(variant))
    assert M.count_parameters(model) / 1e6 == pytest.approx(expected_m, abs=0.02)


def test_forward_shapes_and_deep_supervision():
    model = M.build_model(tiny_spec(), deep_supervision=True).train()
    outs = model(torch.randn(2, 1, 64, 64))
    assert [tuple(o.shape[-2:]) for o in outs] == [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
    model.eval()
    with torch.no_grad():
        single = model(torch.randn(1, 1, 64, 64))
    assert isinstance(single, torch.Tensor) and single.shape == (1, 1, 64, 64)


def test_input_side_must_be_multiple_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        M.build_model(tiny_spec()).eval()(torch.randn(1, 1, 70, 70))


def test_odd_kernel_required():
    with pytest.raises(ValueError, match="odd"):
        M.MedNeXtSpec(kernel_size=4)


def test_spec_round_trip_and_unknown_variant():
    spec = tiny_spec(grn=True, drop_path_rate=0.1)
    assert M.MedNeXtSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        M.MedNeXtSpec.from_variant("XL")


def test_strip_heads_then_strict_load():
    spec = tiny_spec(grn=True)
    trained = M.build_model(spec, deep_supervision=True)
    stripped = M.strip_deep_supervision(trained.state_dict())
    assert not any(k.startswith(("out_1", "out_2", "out_3", "out_4")) for k in stripped)
    M.build_model(spec).load_state_dict(stripped, strict=True)


def test_upkern_resamples_3_to_5():
    small = M.build_model(tiny_spec(kernel_size=3))
    big = M.build_model(tiny_spec(kernel_size=5))
    report = M.upkern_transfer(big, small.state_dict())
    assert report["resampled"] and not report["skipped"]
    key = "enc_block_0.0.conv1.weight"
    assert big.state_dict()[key].shape[-1] == 5
    # a constant 3x3 kernel stays constant when resampled
    const = {k: v.clone() for k, v in small.state_dict().items()}
    const[key] = torch.ones_like(const[key])
    M.upkern_transfer(big, const)
    assert torch.allclose(big.state_dict()[key], torch.ones_like(big.state_dict()[key]), atol=1e-5)


def test_grad_checkpoint_matches_plain_gradients():
    torch.manual_seed(0)
    spec = tiny_spec()
    a = M.build_model(spec, deep_supervision=True).train()
    b = M.build_model(spec, deep_supervision=True, grad_checkpoint=True).train()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 1, 32, 32)
    sum(o.mean() for o in a(x)).backward()
    sum(o.mean() for o in b(x)).backward()
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5)


def test_grn_is_finite_on_large_activations():
    grn = M.GRN(4)
    x = (torch.randn(1, 4, 256, 256) * 300).half()
    assert torch.isfinite(grn(x)).all()


def test_drop_path_identity_in_eval_and_scales_in_train():
    dp = M.DropPath(0.5)
    x = torch.ones(1000, 1)
    dp.eval()
    assert torch.equal(dp(x), x)
    dp.train()
    out = dp(x)
    assert set(out.unique().tolist()) <= {0.0, 2.0}


# ── preprocessing contract ────────────────────────────────────────────────


def test_canvas_is_square_and_pads_bottom_right():
    gray = np.full((100, 200), 120, np.uint8)
    canvas, geom = PRE.to_canvas(gray, 64)
    assert canvas.shape == (64, 64) and (geom.scaled_h, geom.scaled_w) == (32, 64)
    assert canvas[32:].max() == 0 and canvas[:32].min() > 0


def test_roundtrip_geometry_restores_original_size():
    gray = np.random.default_rng(0).integers(0, 255, (90, 150), dtype=np.uint8)
    canvas, geom = PRE.to_canvas(gray, 64)
    back = PRE.canvas_to_original(canvas.astype(np.float32), geom)
    assert back.shape == gray.shape


def test_mask_canvas_follows_image_geometry():
    mask = np.zeros((128, 128), np.uint8)
    mask[40:80, 40:80] = 255
    canvas = PRE.mask_to_canvas(mask, 64)
    assert set(np.unique(canvas)) == {0, 1}
    assert canvas.sum() == pytest.approx(20 * 20, abs=12)


def test_normalise_is_robust_and_deterministic():
    rng = np.random.default_rng(1)
    canvas = np.zeros((64, 64), np.uint8)
    canvas[10:54, 10:54] = rng.integers(40, 160, (44, 44))
    a, b = PRE.normalise(canvas), PRE.normalise(canvas.copy())
    assert np.array_equal(a, b) and a.dtype == np.float32
    inside = a[10:54, 10:54]
    assert abs(inside.mean()) < 0.2 and inside.std() == pytest.approx(1.0, abs=0.2)
    assert np.isfinite(PRE.normalise(np.zeros((64, 64), np.uint8))).all()


def test_normalise_invariant_to_global_intensity_scale():
    rng = np.random.default_rng(2)
    base = np.zeros((64, 64), np.uint8)
    base[8:56, 8:56] = rng.integers(30, 120, (48, 48))
    brighter = np.clip(base.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    assert np.abs(PRE.normalise(base) - PRE.normalise(brighter)).max() < 0.25


def test_decode_matches_file_decoding(tmp_path):
    path = str(tmp_path / "a.jpg")
    cv2.imwrite(path, cv2.cvtColor(np.random.default_rng(3).integers(0, 255, (64, 64), dtype=np.uint8),
                                   cv2.COLOR_GRAY2BGR))
    with open(path, "rb") as handle:
        assert np.array_equal(PRE.decode_gray(handle.read()), PRE.read_gray(path))
    with pytest.raises(ValueError):
        PRE.decode_gray(b"not an image")


def test_to_gray_uint8_handles_rgb_rgba_and_16bit():
    rgb = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert PRE.to_gray_uint8(rgb).shape == (8, 8)
    assert PRE.to_gray_uint8(np.dstack([rgb, np.full((8, 8, 1), 255, np.uint8)])).shape == (8, 8)
    assert PRE.to_gray_uint8(np.full((8, 8), 60000, np.uint16)).dtype == np.uint8


# ── augmentation ──────────────────────────────────────────────────────────


def _aug_case(size=128):
    rng = np.random.default_rng(0)
    head = _head(size, rng)
    mask = np.zeros((size, size), np.uint8)
    mask[50:70, 55:80] = 1
    return head.astype(np.uint8), mask


def test_augment_preserves_shapes_dtypes_and_binary_masks():
    cfg = C.TrainConfig()
    image, mask = _aug_case()
    for k in range(60):
        a, m = AUG.augment(image, mask, np.random.default_rng(k), cfg)
        assert a.shape == image.shape and a.dtype == np.uint8
        assert m.shape == mask.shape and set(np.unique(m)) <= {0, 1}
        assert m.any(), "lesion must stay in frame"


def test_augment_is_reproducible_from_seed():
    cfg = C.TrainConfig()
    image, mask = _aug_case()
    a1, m1 = AUG.augment(image, mask, np.random.default_rng([1, 2, 3]), cfg)
    a2, m2 = AUG.augment(image, mask, np.random.default_rng([1, 2, 3]), cfg)
    assert np.array_equal(a1, a2) and np.array_equal(m1, m2)


def test_augment_actually_changes_samples():
    cfg = C.TrainConfig()
    image, mask = _aug_case()
    changed = sum(
        not np.array_equal(AUG.augment(image, mask, np.random.default_rng(k), cfg)[0], image)
        for k in range(20)
    )
    assert changed >= 18


def test_geometric_transform_moves_image_and_mask_together():
    cfg = C.TrainConfig(aug_hflip_p=0.0, aug_elastic_p=0.0, aug_affine_p=1.0, aug_gamma_p=0.0,
                        aug_bias_field_p=0.0, aug_noise_p=0.0, aug_blur_p=0.0, aug_jpeg_p=0.0,
                        aug_lowres_p=0.0)
    image = np.zeros((128, 128), np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    mask[50:70, 55:80] = 1
    image[mask > 0] = 200
    a, m = AUG.augment(image, mask, np.random.default_rng(5), cfg)
    overlap = ((a > 100) & (m > 0)).sum() / max(1, (m > 0).sum())
    assert overlap > 0.9


# ── losses and metrics ────────────────────────────────────────────────────


def test_dice_loss_extremes():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:10, 4:10] = 1
    perfect = (target * 2 - 1) * 20
    assert L.soft_dice_loss(perfect, target) < 0.05
    assert L.soft_dice_loss(-perfect, target) > 0.8


def test_compound_loss_handles_deep_supervision_and_is_finite():
    crit = L.CompoundLoss()
    target = (torch.rand(2, 1, 32, 32) > 0.8).float()
    outs = [torch.randn(2, 1, s, s, requires_grad=True) for s in (32, 16, 8, 4, 2)]
    loss, parts = crit(outs, target)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"bce", "dice_loss", "total"}
    assert all(o.grad is not None for o in outs)


def test_compound_loss_in_float32_under_half_inputs():
    loss, _ = L.CompoundLoss()(torch.randn(1, 1, 32, 32).half(), torch.zeros(1, 1, 32, 32))
    assert loss.dtype == torch.float32


def test_batch_stats_and_scores():
    probs = torch.zeros(2, 1, 8, 8)
    target = torch.zeros(2, 1, 8, 8)
    probs[0, 0, :4] = 1.0
    target[0, 0, :2] = 1.0
    target[1, 0, :2] = 1.0
    stats = MET.batch_stats(probs, target).numpy()
    scores = MET.scores_from_stats(stats)
    assert scores["dice"][0] == pytest.approx(2 * 16 / (2 * 16 + 16 + 0))
    assert scores["dice"][1] == 0.0 and scores["recall"][1] == 0.0
    both_empty = MET.scores_from_stats(np.zeros((1, 5)))
    assert both_empty["dice"][0] == 1.0


def test_hd95_conventions():
    a = np.zeros((64, 64), bool)
    a[20:30, 20:30] = True
    assert MET.hd95(a, a) == 0.0
    shifted = np.roll(a, 5, axis=1)
    assert MET.hd95(a, shifted) == pytest.approx(5.0, abs=1.5)
    assert np.isnan(MET.hd95(a, np.zeros_like(a)))
    assert MET.hd95(np.zeros_like(a), np.zeros_like(a)) == 0.0


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    ece, *_ = MET.expected_calibration_error(p, y)
    assert ece < 0.01


# ── hygiene ───────────────────────────────────────────────────────────────


def _row(stem, *, sha_image=None, sha_pixels=None, mask="m", inside=1.0, fg=100, std=40.0):
    return {"stem": stem, "ok": True, "shape_match": True, "empty": False, "full": False,
            "gray_std": std, "inside_frac": inside, "fg_px": fg,
            "sha_image": sha_image or stem, "sha_pixels": sha_pixels or stem, "sha_maskpix": mask}


def test_decide_drops_policy():
    group = lambda s: s.split("|")[0]  # noqa: E731
    rows = {
        "L1|a": _row("L1|a", sha_pixels="X", mask="m1", inside=0.99),
        "L1|b": _row("L1|b", sha_pixels="X", mask="m1", inside=0.95),
        "L1|c": _row("L1|c", sha_pixels="Y", mask="m1"),
        "L1|d": _row("L1|d", sha_pixels="Y", mask="m2", inside=0.5),
        "L2|e": _row("L2|e", sha_pixels="Z", mask="m1"),
        "L3|f": _row("L3|f", sha_pixels="Z", mask="m1"),
        "L4|g": _row("L4|g", inside=0.5),
        "L4|h": _row("L4|h", std=2.0),
        "L5|ok": _row("L5|ok"),
    }
    rows["L4|empty"] = {**_row("L4|empty"), "empty": True}
    drops = A.decide_drops(sorted(rows), rows, group, [], align_min=0.9, min_image_std=8.0)
    assert drops["L1|b"]["kept_as"] == "L1|a"                  # same-mask duplicate: keep best aligned
    assert "L1|c" not in drops or "L1|d" in drops             # conflicting masks: one survivor
    assert {"L2|e", "L3|f"} <= set(drops)                       # same image, two lesions: both dropped
    assert "L4|g" in drops and "L4|h" in drops and "L4|empty" in drops
    assert "L5|ok" not in drops and "L1|a" not in drops


def test_near_twins_dropped_only_when_masks_conflict():
    rows = {s: _row(s, mask=m) for s, m in (("a", "m1"), ("b", "m1"), ("c", "m2"))}
    drops = A.decide_drops(["a", "b", "c"], rows, lambda s: "G", [("a", "b"), ("a", "c")],
                           align_min=0.9, min_image_std=8.0)
    assert "b" not in drops and "c" in drops


def test_union_find_and_hamming_pairs():
    uf = A.UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 1)
    assert uf.find(3) == uf.find(0) != uf.find(2)
    hashes = np.array([0b0000, 0b0001, 0b1111_1111], dtype=np.uint64)
    pairs = A.hamming_pairs(hashes, 1)
    assert pairs == [(0, 1, 1)]


def test_openmed_group_key_strips_sequence_and_slice():
    keys = {S.openmed_group_key(s) for s in (
        "T1 - Meningioma frontal 001", "T1C+ - Meningioma frontal 014", "T2 - Meningioma frontal 003")}
    assert keys == {"Meningioma frontal"}
    assert S.openmed_group_key("Germinoma pineal , frontal 002") == "Germinoma pineal , frontal"


def test_audit_detects_planted_defects(tmp_path):
    om, _ = make_sources(tmp_path, n_openmed=60, n_btsc=60, size=64)
    img = os.path.join(om, "images")
    msk = os.path.join(om, "masks")
    names = sorted(os.listdir(img))
    # exact duplicate (same bytes, same mask) of the first file
    base = os.path.splitext(names[0])[0]
    dup = base + " dup"
    open(os.path.join(img, dup + ".jpg"), "wb").write(open(os.path.join(img, names[0]), "rb").read())
    open(os.path.join(msk, dup + ".png"), "wb").write(open(os.path.join(msk, base + ".png"), "rb").read())
    # empty mask
    cv2.imwrite(os.path.join(msk, os.path.splitext(names[1])[0] + ".png"), np.zeros((64, 64), np.uint8))
    # orphan image
    cv2.imwrite(os.path.join(img, "orphan.jpg"), np.full((64, 64), 90, np.uint8))

    cfg = C.TrainConfig(out_dir=str(tmp_path / "out"))
    spec = A.SourceSpec("openmed", om, group_of=S.openmed_group_key, group_kind="lesion")
    audit = A.audit_source(spec, cfg, workers=2, out_dir=cfg.out_dir)
    reasons = " ".join(d["reason"] for d in audit.drops.values())
    assert "empty mask" in reasons and "duplicate image" in reasons
    status = {c.code: c.status for c in audit.checks.checks}
    assert status["H01"] == "WARN" and status["H10"] == "WARN"
    assert not audit.checks.failed
    assert "orphan" not in audit.kept


# ── splits ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    root = tmp_path_factory.mktemp("splitdata")
    om, bt = make_sources(root, size=64)
    cfg = C.TrainConfig(out_dir=str(root / "out"), split_search_tries=30)
    openmed = A.SourceSpec("openmed", om, group_of=S.openmed_group_key, group_kind="lesion")
    btsc = A.SourceSpec("btsc", bt, group_of=None, group_kind="block")
    audits = {s.name: A.audit_source(s, cfg, workers=2, out_dir=cfg.out_dir) for s in (openmed, btsc)}
    return cfg, audits


def test_split_has_no_group_leak_and_covers_all_sources(audited):
    cfg, audits = audited
    manifest = S.build_splits(audits, cfg)
    log = A.CheckLog("t")
    report = S.verify_split(manifest, audits, cfg, log)
    assert report["n_group_leaks"] == 0 and not log.failed
    for split in ("train", "val", "test"):
        assert {r.source for r in manifest.part(split)} == {"openmed", "btsc"}
    stems = [(r.source, r.stem) for r in manifest.all_records()]
    assert len(stems) == len(set(stems))


def test_leak_check_has_a_working_negative_control(audited):
    cfg, audits = audited
    manifest = S.build_splits(audits, cfg)
    victim = manifest.val[0]
    clone = S.Record(**{**victim.__dict__, "stem": victim.stem + "_clone"})
    manifest.train.append(clone)
    log = A.CheckLog("t")
    audits["openmed" if victim.source == "openmed" else "btsc"].rows[clone.stem] = (
        audits[victim.source].rows[victim.stem])
    report = S.verify_split(manifest, audits, cfg, log)
    assert report["n_group_leaks"] >= 1 and log.failed


def test_btsc_blocks_are_contiguous_and_snap_to_dips():
    order = [str(i) for i in range(1, 201)]
    adjacent = np.ones(199)
    adjacent[[37, 71, 110, 150]] = 0.1
    blocks = S.snapped_blocks(order, adjacent, block_size=36, window=8)
    ids = [blocks[s] for s in order]
    assert ids == sorted(ids)
    boundaries = [i + 1 for i in range(199) if ids[i] != ids[i + 1]]
    assert {38, 72, 111}.issubset(set(boundaries))


def test_partition_is_deterministic_and_respects_groups():
    records = [S.Record("s", f"{g}-{i}", "", "", f"g{g}", 0.01 * (g % 5 + 1), 40.0)
               for g in range(40) for i in range(10)]
    a = S.partition_groups(records, 0.1, 0.1, seed=3, tries=40)
    b = S.partition_groups(records, 0.1, 0.1, seed=3, tries=40)
    assert [r.stem for r in a[1]] == [r.stem for r in b[1]]
    groups = [{r.group for r in part} for part in a[:3]]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


# ── export / pipeline ─────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def serving_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    spec = tiny_spec()
    net = M.build_model(spec, deep_supervision=True)
    payload = X.build_serving_checkpoint(
        net.state_dict(), spec, 64,
        postprocess={"threshold": 0.5, "min_area_frac": 0.0, "tta_hflip": True},
        metrics={"val": {"dice": np.float32(0.5)}}, training={"epoch": 3})
    path = str(tmp_path_factory.mktemp("ckpt") / "mednext_brain_seg.pth")
    X.save_serving_checkpoint(payload, path)
    return path


def test_checkpoint_loads_weights_only_and_has_no_aux_heads(serving_checkpoint):
    payload = torch.load(serving_checkpoint, map_location="cpu", weights_only=True)
    assert payload["kind"] == X.MODEL_KIND and payload["format_version"] == 1
    assert not any(k.startswith("out_1") for k in payload["state_dict"])
    assert payload["metrics"]["val"]["dice"] == 0.5


def test_pipeline_segments_any_shape_and_returns_original_size(serving_checkpoint):
    seg = P.MedNeXtSegmenter(serving_checkpoint, device="cpu")
    rng = np.random.default_rng(0)
    for shape in ((64, 64), (90, 150), (200, 120), (33, 47)):
        gray = rng.integers(0, 255, shape, dtype=np.uint8)
        out = seg.predict_gray(gray)
        assert out.mask.shape == shape and out.probability.shape == shape
        assert set(np.unique(out.mask)) <= {0, 1}
        assert 0.0 <= out.probability.min() and out.probability.max() <= 1.0


def test_pipeline_accepts_bytes_and_matches_array_path(serving_checkpoint, tmp_path):
    seg = P.MedNeXtSegmenter(serving_checkpoint, device="cpu")
    gray = np.random.default_rng(1).integers(0, 255, (80, 80), dtype=np.uint8)
    path = str(tmp_path / "x.png")
    cv2.imwrite(path, gray)
    with open(path, "rb") as handle:
        from_bytes = seg.predict(handle.read())
    assert np.allclose(from_bytes.probability, seg.predict_gray(gray).probability, atol=1e-5)


def test_tta_probabilities_are_flip_symmetric_on_square_input(serving_checkpoint):
    seg = P.MedNeXtSegmenter(serving_checkpoint, device="cpu")
    gray = np.random.default_rng(2).integers(0, 255, (64, 64), dtype=np.uint8)
    a = seg.predict_gray(gray, tta=True).probability
    b = seg.predict_gray(gray[:, ::-1].copy(), tta=True).probability
    assert np.allclose(a, b[:, ::-1], atol=1e-4)


def test_postprocess_removes_specks_but_never_empties_a_finding():
    prob = np.zeros((64, 64), np.float32)
    prob[10:30, 10:30] = 0.9
    prob[50, 50] = 0.9
    kept = P.postprocess_mask(prob, 0.5, 0.01)
    assert kept[10:30, 10:30].all() and kept[50, 50] == 0
    only_speck = np.zeros((64, 64), np.float32)
    only_speck[5, 5] = 0.9
    assert P.postprocess_mask(only_speck, 0.5, 0.01).sum() == 1


def test_encoders_return_data_uris():
    gray = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 1
    assert P.encode_mask_png(mask).startswith("data:image/png;base64,")
    assert P.encode_overlay(gray, mask).startswith("data:image/jpeg;base64,")


def test_loader_fails_closed(tmp_path, serving_checkpoint):
    with pytest.raises(P.SegmentationUnavailableError, match="not found"):
        P.MedNeXtSegmenter(str(tmp_path / "missing.pth"))
    empty = tmp_path / "empty.pth"
    empty.write_bytes(b"")
    with pytest.raises(P.SegmentationUnavailableError, match="empty"):
        P.MedNeXtSegmenter(str(empty))
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(P.SegmentationUnavailableError, match="unreadable"):
        P.MedNeXtSegmenter(str(junk))
    raw = tmp_path / "raw.pth"
    torch.save({"model": {}}, raw)
    with pytest.raises(P.SegmentationUnavailableError, match="not a MedNeXt serving"):
        P.MedNeXtSegmenter(str(raw))


def test_loader_rejects_other_preprocessing_version_and_mismatched_weights(tmp_path, serving_checkpoint):
    payload = torch.load(serving_checkpoint, map_location="cpu", weights_only=True)
    bad = {**payload, "preprocess": {**payload["preprocess"], "version": 99}}
    path = tmp_path / "v99.pth"
    torch.save(bad, path)
    with pytest.raises(P.SegmentationUnavailableError, match="preprocessing v99"):
        P.MedNeXtSegmenter(str(path))

    broken = {**payload, "state_dict": {k: v for k, v in list(payload["state_dict"].items())[:-2]}}
    path = tmp_path / "broken.pth"
    torch.save(broken, path)
    with pytest.raises(P.SegmentationUnavailableError, match="do not match"):
        P.MedNeXtSegmenter(str(path))


# ── configuration ─────────────────────────────────────────────────────────


def test_config_roundtrip_and_fingerprint(tmp_path):
    cfg = C.TrainConfig(variant="M", epochs=7)
    path = str(tmp_path / "c.json")
    cfg.save(path)
    import json

    loaded = C.TrainConfig.from_dict(json.load(open(path)))
    assert loaded == cfg and loaded.fingerprint() == cfg.fingerprint()
    assert C.TrainConfig(variant="B").fingerprint() != C.TrainConfig(variant="M").fingerprint()


def test_dataset_discovery_handles_nested_kaggle_mounts(tmp_path):
    om, bt = make_sources(tmp_path / "kaggle" / "input" / "datasets" / "owner", size=32)
    root = tmp_path / "kaggle" / "input"
    os.rename(om, str(root / "datasets" / "owner" / "fypseg"))
    os.rename(bt, str(root / "datasets" / "owner" / "brain-tumor-segmentation"))
    cfg = C.resolve_paths(C.TrainConfig(), input_root=str(root))
    assert cfg.openmed_root.endswith("fypseg") and cfg.btsc_root.endswith("brain-tumor-segmentation")
    assert C.classify_dataset(cfg.btsc_root) == "btsc" and C.classify_dataset(cfg.openmed_root) == "openmed"


def test_dataset_discovery_error_lists_what_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not locate"):
        C.resolve_paths(C.TrainConfig(), input_root=str(tmp_path))


# ── full smoke run ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_smoke_run_end_to_end(tmp_path, sources):
    """Real entry script, real stages, CPU, synthetic data."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(
        sys.modules[M.__name__].__file__)))
    import importlib.util

    script = os.path.join(os.path.dirname(M.__file__), "train_mednext.py")
    spec = importlib.util.spec_from_file_location("train_mednext_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = str(tmp_path / "run")
    code = module.main([
        "--smoke", "--image-size", "64", "--gpus", "0", "--openmed-root", sources[0],
        "--btsc-root", sources[1], "--out", out, "--cache-dir", str(tmp_path / "cache"),
        "--zip-dir", str(tmp_path / "zip"), "--allow-cpu",
    ])
    assert code == 0
    for path in ("best/mednext_brain_seg.pth", "last.pt", "config.json", "metrics/epochs.csv",
                 "metrics/summary.json", "logs/TRAINING_REPORT.txt", "logs/hygiene.json",
                 "splits/split_manifest.csv", "plots/01_learning_curves.png"):
        assert os.path.exists(os.path.join(out, path)), path
    assert any(f.endswith(".zip") for f in os.listdir(tmp_path / "zip"))

    seg = P.MedNeXtSegmenter(os.path.join(out, "best", "mednext_brain_seg.pth"), device="cpu")
    assert seg.postprocess["tuned_on_validation"] is True
    gray = cv2.imread(os.path.join(sources[1], "images", "5.png"), cv2.IMREAD_GRAYSCALE)
    assert seg.predict_gray(gray).mask.shape == gray.shape
