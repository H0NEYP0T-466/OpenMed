"""Stage 5 — plots, human-readable logs and the final ZIP.

Everything written here goes under ``<out_dir>/plots`` and ``<out_dir>/logs``
so the export step can pick up a known layout. Matplotlib is forced to the Agg
backend — Kaggle notebooks have no display, and an accidental ``plt.show()``
would block the cell.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import zipfile
from datetime import datetime, timezone
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("reporting")

COLOURS = {"openmed": "#ed6f5c", "btsc": "#4c9be8", "overall": "#e9b94a"}


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot -> %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(history: list[dict[str, Any]], out_dir: str) -> Optional[str]:
    if not history:
        return None
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], color="#d9534f", lw=1.6)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + Dice (+ IoU head)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [h["val_dice"] for h in history], label="overall", color=COLOURS["overall"], lw=1.8)
    for source, colour in COLOURS.items():
        if source == "overall":
            continue
        series = [(h["per_source"].get(source) or {}).get("dice") for h in history]
        if any(v is not None for v in series):
            axes[1].plot(epochs, series, label=source, color=colour, lw=1.5)
    best_epoch = max(history, key=lambda h: h["val_macro_dice"])["epoch"]
    axes[1].axvline(best_epoch, color="#888", ls="--", lw=1, label=f"best (ep {best_epoch})")
    axes[1].set_title("Validation Dice by source")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    hd = [h["val_hd95_px"] for h in history]
    if any(v is not None for v in hd):
        axes[2].plot(epochs, hd, color="#7b6cf6", lw=1.6)
    axes[2].set_title("Validation HD95 (pixels)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("95th percentile surface distance")
    axes[2].grid(alpha=0.25)

    return _save(fig, os.path.join(_ensure(os.path.join(out_dir, "plots")), "01_learning_curves.png"))


def plot_lr_schedule(csv_path: str, out_dir: str) -> Optional[str]:
    if not os.path.isfile(csv_path):
        return None
    import csv as _csv

    with open(csv_path) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return None

    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(r["lr_encoder"]) for r in rows], label="encoder (TinyViT)", color="#4c9be8")
    ax.plot(epochs, [float(r["lr_decoder"]) for r in rows], label="decoder", color="#ed6f5c")
    ax.set_yscale("log")
    ax.set_title("Learning rate — warmup then cosine")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lr (log)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    return _save(fig, os.path.join(_ensure(os.path.join(out_dir, "plots")), "02_lr_schedule.png"))


def plot_metric_distributions(rows: list[dict[str, Any]], out_dir: str) -> Optional[str]:
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    sources = sorted({r["source"] for r in rows})

    for source in sources:
        values = [r["dice"] for r in rows if r["source"] == source]
        axes[0].hist(values, bins=30, alpha=0.6, label=f"{source} (n={len(values)})",
                     color=COLOURS.get(source, None))
    axes[0].set_title("Per-image Dice (test)")
    axes[0].set_xlabel("Dice")
    axes[0].set_ylabel("images")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    hd = [r["hd95_px"] for r in rows if r.get("hd95_px") is not None and not np.isnan(r["hd95_px"])]
    if hd:
        axes[1].hist(hd, bins=30, color="#7b6cf6", alpha=0.8)
    axes[1].set_title(f"HD95 in pixels (defined for {len(hd)}/{len(rows)})")
    axes[1].set_xlabel("HD95 (px)")
    axes[1].grid(alpha=0.25)

    for source in sources:
        sub = [r for r in rows if r["source"] == source]
        axes[2].scatter([r["gt_fg_frac"] * 100 for r in sub],
                        [r["dice"] for r in sub],
                        s=8, alpha=0.4, label=source, color=COLOURS.get(source, None))
    axes[2].set_title("Dice vs lesion size")
    axes[2].set_xlabel("ground-truth foreground (% of image)")
    axes[2].set_ylabel("Dice")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    return _save(fig, os.path.join(_ensure(os.path.join(out_dir, "plots")), "03_test_distributions.png"))


def plot_calibration(rows: list[dict[str, Any]], out_dir: str) -> Optional[str]:
    """Does the IoU head's predicted score track the achieved IoU?"""
    if not rows:
        return None
    pred = np.array([r["iou_head_pred"] for r in rows], dtype=np.float64)
    actual = np.array([r["iou"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(pred, actual, s=10, alpha=0.45, color="#ed6f5c")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#888", lw=1)
    axes[0].set_title("IoU-head calibration")
    axes[0].set_xlabel("predicted IoU")
    axes[0].set_ylabel("achieved IoU")
    axes[0].grid(alpha=0.25)

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pred, bins) - 1, 0, len(bins) - 2)
    xs, ys = [], []
    for b in range(len(bins) - 1):
        mask = idx == b
        if mask.sum() >= 5:
            xs.append(pred[mask].mean())
            ys.append(actual[mask].mean())
    if xs:
        axes[1].plot(xs, ys, "o-", color="#4c9be8")
        axes[1].plot([0, 1], [0, 1], ls="--", color="#888", lw=1)
    axes[1].set_title("Binned calibration (>=5 samples per bin)")
    axes[1].set_xlabel("mean predicted IoU")
    axes[1].set_ylabel("mean achieved IoU")
    axes[1].grid(alpha=0.25)

    return _save(fig, os.path.join(_ensure(os.path.join(out_dir, "plots")), "04_iou_head_calibration.png"))


def plot_baseline_vs_finetuned(
    baseline: dict[str, Any], summary: dict[str, Any], out_dir: str
) -> Optional[str]:
    if not baseline or not summary:
        return None

    sources = sorted(set(baseline.get("per_source", {})) | set(summary["test"]["per_source"]))
    base_vals = [(baseline["per_source"].get(s) or {}).get("dice") or 0.0 for s in sources]
    fine_vals = [(summary["test"]["per_source"].get(s) or {}).get("dice") or 0.0 for s in sources]

    x = np.arange(len(sources))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - width / 2, base_vals, width, label="pretrained (val)", color="#9aa0a6")
    ax.bar(x + width / 2, fine_vals, width, label="fine-tuned (test)", color="#ed6f5c")
    for i, (b, f) in enumerate(zip(base_vals, fine_vals)):
        ax.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.0)
    ax.set_title("Pretrained baseline vs fine-tuned")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.text(
        0.99, 0.02,
        "baseline is val, fine-tuned is test — not the same split",
        transform=ax.transAxes, ha="right", fontsize=7, color="#888",
    )
    return _save(fig, os.path.join(_ensure(os.path.join(out_dir, "plots")), "05_baseline_vs_finetuned.png"))


def plot_qualitative(
    cfg: Any,
    records: list[Any],
    out_dir: str,
    *,
    count: int = 8,
    seed: int = 0,
) -> Optional[str]:
    """Input / ground truth / prediction overlays from the best checkpoint.

    Accepts ``Record`` objects or the dict form written by
    ``SplitManifest.save``. Torch is imported lazily so this module stays
    importable on a machine without a GPU.
    """
    import cv2
    import torch

    from .data import Record, TumourSegDataset
    from .medsam_model import build_medsam_lite
    from .preprocessor import MEDSAM_INPUT_SIZE, postprocess_mask

    best = os.path.join(cfg.out_dir, "best", "lite_medsam.pth")
    if not os.path.isfile(best):
        logger.warning("No best checkpoint at %s — skipping qualitative grid", best)
        return None

    def as_record(row: Any) -> Record:
        if isinstance(row, Record):
            return row
        return Record(
            source=row["source"],
            stem=row["stem"],
            image_path=row.get("image_path") or row["image"],
            mask_path=row.get("mask_path") or row["mask"],
            group=row["group"],
            group_is_subject_verified=row.get("group_is_subject_verified", False),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_medsam_lite()
    model.load_state_dict(torch.load(best, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()

    chosen = list(records)
    if len(chosen) > count:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(chosen), size=count, replace=False)
        chosen = [chosen[int(i)] for i in idx]

    dataset = TumourSegDataset([as_record(r) for r in chosen], cfg, train=False, jitter=0)

    columns = 4
    rows = int(np.ceil(len(dataset) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        box = sample["box"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(image, boxes=box)

        native = sample["native_mask"].numpy().astype(np.uint8)
        valid_hw = sample["valid_hw"].numpy()
        pred = postprocess_mask(
            logits.float().cpu(),
            original_hw=(int(native.shape[0]), int(native.shape[1])),
            resized_hw=(int(valid_hw[0]), int(valid_hw[1])),
            target_size=MEDSAM_INPUT_SIZE,
        )
        pred_bin = pred > 0
        gt_bin = native > 0
        inter = int((pred_bin & gt_bin).sum())
        union = int((pred_bin | gt_bin).sum())
        dice = (2 * inter / (int(pred_bin.sum()) + int(gt_bin.sum()))) if (pred_bin.sum() + gt_bin.sum()) else 1.0

        base = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        overlay = base.copy()
        overlay[gt_bin] = (overlay[gt_bin] * 0.45 + np.array([60, 220, 90]) * 0.55).astype(np.uint8)
        overlay[pred_bin] = (overlay[pred_bin] * 0.45 + np.array([240, 110, 92]) * 0.55).astype(np.uint8)
        overlay = cv2.addWeighted(base, 0.45, overlay, 0.55, 0)

        ax = axes[i]
        ax.imshow(overlay)
        ax.set_title(
            f"{sample['source']} · dice {dice:.3f}\n{sample['stem'][:34]}",
            fontsize=7,
        )
        ax.axis("off")

    for j in range(len(dataset), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Qualitative test predictions — green = ground truth, coral = prediction, "
        "yellow where they agree",
        fontsize=9,
    )
    return _save(fig, os.path.join(_ensure(os.path.join(out_dir, "plots")), "06_qualitative.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    cfg: Any,
    out_dir: str,
    *,
    env: dict[str, Any],
    weights_info: dict[str, Any],
    audit_reports: dict[str, Any],
    split_report: dict[str, Any],
    cross_source: dict[str, Any],
    summary: Optional[dict[str, Any]],
) -> str:
    """Write ``logs/TRAINING_REPORT.txt`` — the artefact for the SRS."""
    lines: list[str] = []
    add = lines.append

    def rule(char: str = "=") -> None:
        add(char * 78)

    def section(title: str) -> None:
        add("")
        rule()
        add(title)
        rule()

    add("OpenMed — LiteMedSAM segmentation fine-tuning")
    add(f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC")
    add(f"host {platform.platform()}")

    section("1. RUN CONFIGURATION")
    for key, value in sorted(cfg.to_dict().items()):
        add(f"  {key:<32} {value}")

    section("2. ENVIRONMENT")
    add(f"  python                {env['python']}")
    add(f"  torch                 {env['torch']}  (cuda {env['cuda_version']})")
    add(f"  cuda available        {env['cuda_available']}   gpus {env['gpu_count']}")
    for gpu in env["gpus"]:
        add(f"    gpu {gpu['index']}               {gpu['name']}  {gpu['vram_gb']} GB  sm_{gpu['capability']}")
    for path, usage in env["disk"].items():
        add(f"  disk {path:<16} {usage['free_gb']} GB free / {usage['total_gb']} GB")

    section("3. PRETRAINED WEIGHTS")
    add(f"  source                {weights_info.get('source')}")
    add(f"  path                  {weights_info.get('path')}")
    add(f"  sha256                {weights_info.get('sha256')}")
    add(f"  size                  {weights_info.get('size_bytes', 0) / 1024**2:.1f} MB")
    add(f"  parameters            {weights_info.get('parameters_millions')} M")
    add(f"  strict load + forward {weights_info.get('forward_ok')}")
    add("  note                  same Google Drive release the serving backend uses,")
    add("                        so training and inference start from identical weights")

    section("4. DATASET HYGIENE")
    for name, report in audit_reports.items():
        add(f"  ── {name} ──")
        add(f"  paired stems          {report['n_stems']}")
        orphan = report["orphans"]
        add(f"  images without mask   {len(orphan['images_without_mask'])}")
        add(f"  masks without image   {len(orphan['masks_without_image'])}")
        geom = report["geometry"]
        add(f"  unreadable            {len(geom['unreadable'])}")
        add(f"  shape mismatches      {len(geom['shape_mismatch'])}")
        add(f"  non-binary masks      {len(geom['non_binary'])}")
        add(f"  empty / full masks    {len(geom['empty_masks'])} / {len(geom['full_masks'])}")
        add(f"  misaligned (<{cfg.align_min_inside:.0%} on anatomy)  {geom['misaligned_count']}")
        fg = geom["fg_frac"]
        add(f"  foreground fraction   min {fg['min']:.4f}  median {fg['median']:.4f}  max {fg['max']:.4f}")
        add(f"  resolutions           {geom['resolutions']}")
        dup = report["duplicates"]
        add(f"  duplicate image sets  {dup['image_duplicate_groups']} groups / {dup['image_duplicate_files']} files")
        add(f"    same mask           {dup['image_dup_groups_same_mask']} groups")
        add(f"    DIFFERENT mask      {dup['image_dup_groups_conflicting_mask']} groups / "
            f"{dup['image_dup_files_conflicting_mask']} files")
        nd = report["near_duplicates"]
        top = nd["counts"][f"hamming<={nd['hamming_max']}"]
        add(f"  pHash <= {nd['hamming_max']}            within-group {top['within_group']}  "
            f"cross-group {top['cross_group']}  (over {nd['sampled']} images)")
        for problem in report["problems"]:
            add(f"  PROBLEM               {problem}")
        for note in report["notes"]:
            add(f"  note                  {note}")

    section("5. CROSS-SOURCE OVERLAP")
    add(f"  identical image bytes {cross_source.get('exact_sha256_overlap')}")
    add(f"  {cross_source.get('exact_note', '')}")
    add(f"  perceptual candidates {len(cross_source.get('candidates', []))}")
    add(f"  confirmed shared      {cross_source.get('confirmed_shared_images')}")
    for cand in cross_source.get("candidates", [])[:10]:
        add(f"    hamming {cand['hamming']}  pixel MAE {cand.get('pixel_mae', float('nan')):.2f}  "
            f"{cand['a'][:26]} ~ {cand['b'][:26]}")

    section("6. SPLIT")
    for split, counts in split_report["counts"].items():
        add(f"  {split:<20} {counts}   total {sum(counts.values())}")
    add(f"  groups across splits  {split_report['n_group_leaks']}  "
        f"(0 means no group appears twice)")
    add(f"  group keys            openmed={split_report['group_kind']['openmed']}")
    add(f"                        btsc={split_report['group_kind']['btsc']}")
    exposure = split_report.get("btsc_boundary_exposure", {})
    if exposure.get("applicable"):
        add("")
        add("  BTSC residual adjacency risk (no patient identifier exists in the PNG export):")
        add(f"    btsc slices                        {exposure['btsc_total_slices']}")
        add(f"    adjacent pairs crossing a split    {exposure['adjacent_pairs_crossing_a_split']}")
        add(f"    slices touching such a pair        {exposure['slices_touching_a_cross_split_pair']}")
        add(f"    exposed fraction                   {exposure['exposed_fraction']}")
        add(f"    {exposure['interpretation']}")
    add("")
    add(f"  CAVEAT: {split_report['caveat']}")

    section("7. TRAINING")
    if summary:
        add(f"  epochs run            {summary['epochs_run']}")
        add(f"  best val macro Dice   {summary['best_val_macro_dice']:.4f}")
        add("")
        add("  Test — ground-truth-derived box prompt:")
        _write_metrics(add, summary["test"])
        add("")
        add(f"  Test — perturbed prompt (+/-{summary['test_perturbed_prompt']['jitter_px']} px):")
        _write_metrics(add, summary["test_perturbed_prompt"])
        add("")
        add(f"  {summary['metric_notes']['prompt']}")
        add(f"  HD95 units: {summary['metric_notes']['hd95_units']}; "
            f"undefined when {summary['metric_notes']['hd95_undefined']}.")
    else:
        add("  training did not complete — no summary available")

    section("8. ARTEFACTS")
    add("  best/lite_medsam.pth        bare state dict — drop-in for the serving checkpoint")
    add("  last.pt                     resumable (optimizer, scheduler, scaler, RNG)")
    add("  metrics/epochs.csv          one row per epoch")
    add("  metrics/history.json        per-epoch curves")
    add("  metrics/summary.json        final test metrics")
    add("  metrics/test_per_image.json per-image test metrics")
    add("  metrics/baseline_val.json   pretrained model, before training")
    add("  splits/{train,val,test}.json")
    add("  plots/*.png                 learning curves, calibration, qualitative")
    add("  rank*.log                   per-process logs")

    section("9. LIMITATIONS — quote these, do not omit them")
    add("  1. Prompt is not end-to-end. Boxes come from the ground-truth mask, not")
    add("     from Grad-CAM. These numbers measure conditional segmentation quality.")
    add("  2. HD95 is in pixels. No validated pixel spacing exists for these slices,")
    add("     so no millimetre figure is reported.")
    add("  3. BTSC has no patient identifier in the PNG export. Its split uses")
    add("     contiguous blocks; residual boundary exposure is reported above.")
    add("  4. OpenMed group keys are lesion keys derived from filenames, not verified")
    add("     patient identifiers.")
    add("  5. Only two sources. Both appear in train and test, but neither is a")
    add("     genuinely external cohort.")
    add("  6. Validation is used for early stopping and model selection; the test")
    add("     split is evaluated once, on the best checkpoint.")

    text = "\n".join(lines) + "\n"
    path = os.path.join(_ensure(os.path.join(out_dir, "logs")), "TRAINING_REPORT.txt")
    with open(path, "w") as fh:
        fh.write(text)
    logger.info("report -> %s", path)
    return path


def _write_metrics(add, block: dict[str, Any]) -> None:
    overall = block.get("overall", {})
    add(f"    overall   n={overall.get('n')}  Dice={_fmt(overall.get('dice'))}  "
        f"IoU={_fmt(overall.get('iou'))}  P={_fmt(overall.get('precision'))}  "
        f"R={_fmt(overall.get('recall'))}  HD95={_fmt(overall.get('hd95_px'))}px")
    for source, values in block.get("per_source", {}).items():
        add(f"    {source:<9} n={values.get('n')}  Dice={_fmt(values.get('dice'))}  "
            f"IoU={_fmt(values.get('iou'))}  HD95={_fmt(values.get('hd95_px'))}px  "
            f"(HD95 defined for {values.get('hd95_defined')})")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_zip(out_dir: str, working_dir: str = "/kaggle/working") -> str:
    """Bundle the best model, plots, logs and metrics into one ZIP."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    zip_path = os.path.join(working_dir, f"openmed_litemedsam_{stamp}.zip")

    include_roots = ("best", "plots", "logs", "metrics", "splits")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root in include_roots:
            base = os.path.join(out_dir, root)
            if not os.path.isdir(base):
                continue
            for dirpath, _dirnames, filenames in os.walk(base):
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    archive.write(full, os.path.relpath(full, out_dir))

        for name in ("config.json", "last.pt"):
            full = os.path.join(out_dir, name)
            if os.path.isfile(full):
                archive.write(full, name)

    size_mb = os.path.getsize(zip_path) / 1024**2
    logger.info("export -> %s (%.1f MB)", zip_path, size_mb)
    return zip_path
