#!/usr/bin/env python
"""OpenMed — LiteMedSAM fine-tuning on Kaggle. Single entry point.

Run this file and the whole pipeline runs in order:

    !python trainKaggle.py

Stages
------
  0  configuration and logging
  1  environment, dependencies, pretrained weights (Google Drive, verified)
  2  dataset hygiene audit on both sources
  3  cross-source overlap check
  4  split construction (grouped, with leakage verification)
  5  DDP training on 2 x T4, early stopping at 10 epochs
  6  final test evaluation + plots
  7  report and ZIP export to /kaggle/working

Expected mounts
---------------
    /kaggle/input/fypseg                     OpenMed segmentation (10,056 pairs)
    /kaggle/input/brain-tumor-segmentation   figshare/BTSC (3,064 pairs)

Both are auto-detected; override with --openmed-root / --btsc-root.

Every stage is also importable on its own, so a failed run can be debugged
stage by stage instead of from the top.

Usage examples
--------------
    !python trainKaggle.py                          # full run
    !python trainKaggle.py --epochs 30 --batch 8
    !python trainKaggle.py --audit-only             # hygiene report, no training
    !python trainKaggle.py --resume /kaggle/working/run/last.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# The package lives next to this file; Kaggle runs the notebook from
# /kaggle/working, so make the import path explicit rather than relying on cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from medsam_training import audit as audit_mod
from medsam_training import bootstrap, config as C
from medsam_training import data as data_mod
from medsam_training import engine, reporting

logger = logging.getLogger("trainKaggle")


# ─────────────────────────────────────────────────────────────────────────────
# Source layout detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_layout(root: str, name: str) -> dict[str, str]:
    """Work out where images and masks live inside a mounted dataset.

    Kaggle mirrors vary: some keep ``images/`` + ``masks/``, some keep the
    upstream figshare export with ``<stem>.png`` + ``<stem>_mask.png`` in one
    directory. Rather than guess, probe the known shapes first, then fall back
    to scanning for whichever pairing actually exists on disk.
    """
    candidates = [
        ("images", "masks", ""),
        ("images", "masks", "_mask"),
        ("png_dataset", "png_dataset", "_mask"),
        (".", ".", "_mask"),
        ("images", "labels", ""),
    ]

    def usable(images_dir: str, masks_dir: str, suffix: str) -> int:
        images_path = os.path.join(root, images_dir)
        masks_path = os.path.join(root, masks_dir)
        if not (os.path.isdir(images_path) and os.path.isdir(masks_path)):
            return 0
        spec = audit_mod.SourceSpec(
            name=name, root=root,
            images_dir=images_dir, masks_dir=masks_dir, mask_suffix=suffix,
        )
        try:
            return len(audit_mod.list_stems(spec))
        except OSError:
            return 0

    for images_dir, masks_dir, suffix in candidates:
        count = usable(images_dir, masks_dir, suffix)
        if count >= 50:
            logger.info(
                "[%s] layout: images=%s  masks=%s  mask_suffix=%r  (%d paired stems)",
                name, images_dir, masks_dir, suffix, count,
            )
            return {"images_dir": images_dir, "masks_dir": masks_dir, "mask_suffix": suffix}

    # Nothing matched a known shape. Scan one level down for whatever pairing
    # does exist, so an unfamiliar mirror still works without a code change.
    best: tuple[int, dict[str, str]] | None = None
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        entries = []

    for entry in entries:
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        for suffix in ("_mask", ""):
            count = usable(entry, entry, suffix)
            if count >= 50 and (best is None or count > best[0]):
                best = (count, {"images_dir": entry, "masks_dir": entry, "mask_suffix": suffix})

    if best is not None:
        layout = best[1]
        logger.info(
            "[%s] layout found by scan: %s/%s  mask_suffix=%r  (%d paired stems)",
            name, layout["images_dir"], layout["masks_dir"], layout["mask_suffix"], best[0],
        )
        return layout

    listing = {
        entry: (len(os.listdir(os.path.join(root, entry)))
                if os.path.isdir(os.path.join(root, entry)) else "file")
        for entry in entries
    }
    raise RuntimeError(
        f"Could not determine the layout of {name} at {root}.\n"
        f"  Top level: {listing}\n"
        "  Expected one of: images/ + masks/ · png_dataset/ with <stem>_mask.png ·\n"
        "  a single folder of <stem>.png + <stem>_mask.png."
    )


def openmed_group_lookup(root: str):
    """Return a stem -> group callable for OpenMed.

    Prefers ``manifest.csv`` when the dataset ships one, because that is the
    grouping the cleaning pass actually used. Falls back to deriving the key
    from the filename, which reproduces the same key for every stem seen so far.
    """
    manifest = os.path.join(root, "manifest.csv")
    if os.path.isfile(manifest):
        import csv

        mapping: dict[str, str] = {}
        try:
            with open(manifest, newline="") as fh:
                for row in csv.DictReader(fh):
                    stem = (row.get("stem") or "").strip()
                    group = (row.get("group") or "").strip()
                    if stem and group:
                        mapping[stem] = group
        except (OSError, csv.Error) as exc:
            logger.warning("manifest.csv present but unreadable (%s) — using filename keys", exc)

        if mapping:
            logger.info("[openmed] group keys from manifest.csv (%d entries)", len(mapping))

            def from_manifest(stem: str) -> str:
                return mapping.get(stem, data_mod.openmed_group_key(stem))

            from_manifest.__name__ = "openmed_group_from_manifest"
            return from_manifest

    logger.info("[openmed] no manifest.csv — group keys derived from filenames")

    def from_filename(stem: str) -> str:
        return data_mod.openmed_group_key(stem)

    from_filename.__name__ = "openmed_group_from_filename"
    return from_filename


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = C.TrainConfig()
    parser = argparse.ArgumentParser(
        description="Fine-tune LiteMedSAM on the combined OpenMed + BTSC segmentation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--openmed-root", default=None, help="OpenMed segmentation dataset root")
    parser.add_argument("--btsc-root", default=None, help="BTSC segmentation dataset root")
    parser.add_argument("--out", default=defaults.out_dir, help="output directory")
    parser.add_argument("--pretrained", default=None, help="existing lite_medsam.pth (skips download)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch", type=int, default=defaults.batch_size, help="per GPU")
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--encoder-lr", type=float, default=defaults.encoder_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val-frac", type=float, default=defaults.val_frac)
    parser.add_argument("--test-frac", type=float, default=defaults.test_frac)
    parser.add_argument("--btsc-block-size", type=int, default=defaults.btsc_block_size)
    parser.add_argument("--gpus", type=int, default=defaults.gpus, help="1 or 2")
    parser.add_argument("--resume", default=None, help="resume from a last.pt")
    parser.add_argument("--grad-checkpoint", action="store_true", help="trade speed for VRAM")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="run hygiene + split, then stop")
    parser.add_argument(
        "--force-audit", action="store_true",
        help="re-run the hygiene audit even when a cached audit.json exists",
    )
    parser.add_argument(
        "--skip-audit", action="store_true",
        help="reuse the cached audit if present (this is the default)",
    )
    parser.add_argument("--no-export", action="store_true", help="skip the final ZIP")
    parser.add_argument("--smoke", action="store_true", help="2 steps per epoch, for wiring checks only")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> C.TrainConfig:
    cfg = C.TrainConfig(
        openmed_root=args.openmed_root,
        btsc_root=args.btsc_root,
        out_dir=args.out,
        pretrained=args.pretrained,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        btsc_block_size=args.btsc_block_size,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
        workers=args.workers,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        gpus=args.gpus,
        resume=args.resume,
        grad_checkpoint=args.grad_checkpoint,
        amp=not args.no_amp,
        smoke=args.smoke,
    )
    if cfg.smoke:
        cfg.epochs = 2
        cfg.max_steps_per_epoch = 2
        cfg.patience = 2
        cfg.batch_size = 2
        cfg.workers = 0
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-13s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(cfg.out_dir, "logs", "run.log")),
        ],
        force=True,
    )

    started = time.time()

    # ── stage 1: environment and weights ─────────────────────────────────
    logger.info("STAGE 1/7 — environment, dependencies, pretrained weights")
    versions = bootstrap.check_dependencies()
    env = bootstrap.environment_report()
    bootstrap.log_environment(env)
    logger.info("dependency versions: %s", versions)
    bootstrap.assert_training_hardware(env, cfg.gpus)

    cfg = C.resolve_paths(cfg)
    logger.info("openmed root : %s", cfg.openmed_root)
    logger.info("btsc root    : %s", cfg.btsc_root)

    _, weights_info = bootstrap.prepare_pretrained(cfg)
    logger.info(
        "pretrained ready: %s (%.1f MB, %s M params, sha256 %s)",
        weights_info["path"], weights_info["size_bytes"] / 1024**2,
        weights_info["parameters_millions"], weights_info["sha256"][:16],
    )

    # ── stage 2: dataset hygiene ─────────────────────────────────────────
    logger.info("STAGE 2/7 — dataset hygiene audit")
    om_layout = detect_layout(cfg.openmed_root, "openmed")
    bt_layout = detect_layout(cfg.btsc_root, "btsc")

    openmed = audit_mod.SourceSpec(
        name="openmed",
        root=cfg.openmed_root,
        group_of=openmed_group_lookup(cfg.openmed_root),
        group_kind="lesion",
        group_is_subject_verified=False,
        **om_layout,
    )
    btsc = audit_mod.SourceSpec(
        name="btsc",
        root=cfg.btsc_root,
        group_of=None,
        group_kind="none",
        group_is_subject_verified=False,
        **bt_layout,
    )

    audit_cache = os.path.join(cfg.out_dir, "audit.json")
    # The audit is a one-time gate, not a per-run cost. Re-running it on every
    # restart costs ~6 minutes of GPU-idle time and changes nothing, because it
    # only reads the datasets. Reuse the cached result unless explicitly asked
    # to redo it.
    reuse_audit = os.path.isfile(audit_cache) and not args.force_audit

    if reuse_audit:
        import json

        logger.info("STAGE 2-3/7 — reusing cached audit (pass --force-audit to redo)")
        with open(audit_cache) as fh:
            cached = json.load(fh)
        openmed_report = cached["openmed"]
        btsc_report = cached["btsc"]
        cross_source = cached["cross_source"]
        logger.info(
            "  cached: openmed %d stems, btsc %d stems, cross-source overlap %d",
            openmed_report["n_stems"], btsc_report["n_stems"],
            cross_source.get("confirmed_shared_images", 0),
        )
    else:
        if os.path.isfile(audit_cache):
            logger.info("STAGE 2/7 — --force-audit: re-running the hygiene audit")
        else:
            logger.info("STAGE 2/7 — no cached audit found: running the hygiene audit")
        openmed_report = audit_mod.audit_source(
            openmed,
            workers=cfg.workers or 2,
            align_min_inside=cfg.align_min_inside,
            phash_hamming_max=cfg.phash_hamming_max,
            phash_sample=cfg.phash_sample,
            seed=cfg.seed,
        ).to_dict()
        btsc_report = audit_mod.audit_source(
            btsc,
            workers=cfg.workers or 2,
            align_min_inside=cfg.align_min_inside,
            phash_hamming_max=cfg.phash_hamming_max,
            phash_sample=cfg.phash_sample,
            seed=cfg.seed,
        ).to_dict()

        # ── stage 3: cross-source overlap ────────────────────────────────
        logger.info("STAGE 3/7 — cross-source overlap")
        cross_source = audit_mod.cross_source_overlap(
            openmed, btsc, workers=cfg.workers or 2,
            hamming_max=cfg.phash_hamming_max, seed=cfg.seed,
        )
        logger.info(
            "overlap: %d exact-byte matches, %d perceptual candidates, %d confirmed shared",
            cross_source["exact_sha256_overlap"],
            len(cross_source["candidates"]),
            cross_source["confirmed_shared_images"],
        )

        import json

        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(audit_cache, "w") as fh:
            json.dump(
                {"openmed": openmed_report, "btsc": btsc_report, "cross_source": cross_source},
                fh, indent=2,
            )

    for report in (openmed_report, btsc_report):
        if report["problems"]:
            logger.warning("[%s] %d hygiene problem(s):", report["source"], len(report["problems"]))
            for problem in report["problems"]:
                logger.warning("    - %s", problem)
        else:
            logger.info("[%s] hygiene: no problems found", report["source"])

    # ── stage 4: split ───────────────────────────────────────────────────
    logger.info("STAGE 4/7 — split construction")
    manifest = data_mod.build_splits(
        openmed, btsc,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        btsc_block_size=cfg.btsc_block_size,
        seed=cfg.seed,
    )
    split_dir = os.path.join(cfg.out_dir, "splits")
    manifest.save(split_dir)
    split_report = manifest.report

    for split, counts in split_report["counts"].items():
        logger.info("  %-6s %s  (total %d)", split, counts, sum(counts.values()))
    if split_report["n_group_leaks"]:
        raise RuntimeError(
            f"{split_report['n_group_leaks']} group(s) appear in more than one split — "
            "refusing to train on a leaking split. See splits in " + cfg.out_dir
        )
    logger.info("  group leakage: 0 (no group appears in more than one split)")
    exposure = split_report["btsc_boundary_exposure"]
    if exposure.get("applicable"):
        logger.warning(
            "  BTSC has no patient id: %d slices (%.2f%%) sit at a block boundary and may "
            "share a scan across splits. Reported, not hidden.",
            exposure["slices_touching_a_cross_split_pair"],
            100 * (exposure["exposed_fraction"] or 0.0),
        )

    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    if args.audit_only:
        logger.info("--audit-only: stopping before training. Artefacts in %s", cfg.out_dir)
        reporting.write_report(
            cfg, cfg.out_dir, env=env, weights_info=weights_info,
            audit_reports={"openmed": openmed_report, "btsc": btsc_report},
            split_report=split_report, cross_source=cross_source, summary=None,
        )
        return 0

    # ── stage 5: training ────────────────────────────────────────────────
    logger.info("STAGE 5/7 — DDP training on %d GPU(s)", cfg.gpus)
    world_size = cfg.gpus
    import torch.multiprocessing as mp

    mp.spawn(engine.run_worker, nprocs=world_size, args=(world_size, cfg.to_dict()), join=True)

    # ── stage 6: plots ───────────────────────────────────────────────────
    logger.info("STAGE 6/7 — plots")
    import json

    history_path = os.path.join(cfg.out_dir, "metrics", "history.json")
    summary_path = os.path.join(cfg.out_dir, "metrics", "summary.json")
    baseline_path = os.path.join(cfg.out_dir, "metrics", "baseline_val.json")
    test_path = os.path.join(cfg.out_dir, "metrics", "test_per_image.json")

    def _load(path, default):
        """Read JSON if present, else the default. Closes the handle."""
        if not os.path.isfile(path):
            return default
        with open(path) as fh:
            return json.load(fh)

    history = _load(history_path, [])
    summary = _load(summary_path, None)
    baseline = _load(baseline_path, {})
    test_rows = _load(test_path, [])

    reporting.plot_learning_curves(history, cfg.out_dir)
    reporting.plot_lr_schedule(os.path.join(cfg.out_dir, "metrics", "epochs.csv"), cfg.out_dir)
    reporting.plot_metric_distributions(test_rows, cfg.out_dir)
    reporting.plot_calibration(test_rows, cfg.out_dir)
    if summary:
        reporting.plot_baseline_vs_finetuned(baseline, summary, cfg.out_dir)
    reporting.plot_qualitative(cfg, manifest.test, cfg.out_dir)

    # ── stage 7: report and export ───────────────────────────────────────
    logger.info("STAGE 7/7 — report and export")
    reporting.write_report(
        cfg, cfg.out_dir, env=env, weights_info=weights_info,
        audit_reports={"openmed": openmed_report, "btsc": btsc_report},
        split_report=split_report, cross_source=cross_source, summary=summary,
    )

    if not args.no_export:
        zip_path = reporting.export_zip(cfg.out_dir, os.path.dirname(cfg.out_dir.rstrip("/")) or "/kaggle/working")
        logger.info("ZIP ready: %s", zip_path)

    elapsed = time.time() - started
    logger.info("DONE in %.1f min. Artefacts under %s", elapsed / 60, cfg.out_dir)
    if summary:
        logger.info(
            "FINAL — best val macro Dice %.4f | test Dice %.4f | test HD95 %.1f px",
            summary["best_val_macro_dice"],
            summary["test"]["overall"].get("dice") or float("nan"),
            summary["test"]["overall"].get("hd95_px") or float("nan"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
