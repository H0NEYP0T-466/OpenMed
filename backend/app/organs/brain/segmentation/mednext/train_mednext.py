#!/usr/bin/env python
"""OpenMed — MedNeXt-2D brain-lesion segmentation. Single Kaggle entry point.

    !git clone <repo> && cd <repo>
    !python backend/app/organs/brain/segmentation/mednext/train_mednext.py

Stages
------
  1  environment, dependencies, hardware assertion
  2  dataset hygiene audit of both sources (H01-H14) and the cleaning decision
  3  split construction (grouped / snapped blocks) and split verification (V1-V8)
  4  array cache (shared preprocessor) and cache verification (H15-H16)
  5  DDP training: AdamW, EMA, augmentation, early stopping at patience 10
  6  post-processing search on validation, one pass over the test split through
     the serving pipeline, export of the self-describing checkpoint (H17-H18)
  7  plots, TRAINING_REPORT.txt and a ZIP in /kaggle/working

Expected mounts (auto-detected, override with --openmed-root / --btsc-root):
    /kaggle/input/.../fypseg                     OpenMed segmentation
    /kaggle/input/.../brain-tumor-segmentation   figshare / BTSC

Deployment: copy ``best/mednext_brain_seg.pth`` (also copied next to the ZIP) to
``backend/app/organs/brain/segmentation/mednext/checkpoints/``. Nothing else.

    !python .../train_mednext.py --smoke --limit 120        # wiring check, minutes
    !python .../train_mednext.py --audit-only               # hygiene + split only
    !python .../train_mednext.py --variant M --epochs 120
    !python .../train_mednext.py --resume /kaggle/working/mednext_run/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from mednext.model import MedNeXtSpec, build_model, count_parameters, spec_summary  # noqa: E402
from mednext.training import audit as audit_mod  # noqa: E402
from mednext.training import bootstrap, engine, evaluate, reporting, splits  # noqa: E402
from mednext.training import config as C  # noqa: E402
from mednext.training import data as data_mod  # noqa: E402
from mednext.training.export import plain  # noqa: E402

logger = logging.getLogger("train_mednext")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = C.TrainConfig()
    p = argparse.ArgumentParser(
        description="Train MedNeXt-2D for brain-lesion segmentation (OpenMed + BTSC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--openmed-root")
    p.add_argument("--btsc-root")
    p.add_argument("--out", default=d.out_dir)
    p.add_argument("--cache-dir")
    p.add_argument("--zip-dir", default=d.zip_dir)
    p.add_argument("--variant", default=d.variant, choices=["S", "B", "M", "L"])
    p.add_argument("--kernel-size", type=int, default=d.kernel_size, choices=[3, 5, 7])
    p.add_argument("--grn", action="store_true", help="add Global Response Normalisation")
    p.add_argument("--drop-path", type=float, default=d.drop_path)
    p.add_argument("--no-deep-supervision", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true", help="trade speed for VRAM")
    p.add_argument("--upkern-from", help="k=3 serving checkpoint to initialise a larger kernel from")
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch", type=int, default=d.batch_size, help="per GPU")
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--ema-decay", type=float, default=d.ema_decay)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--val-frac", type=float, default=d.val_frac)
    p.add_argument("--test-frac", type=float, default=d.test_frac)
    p.add_argument("--btsc-mode", default=d.btsc_mode, choices=["block", "train_only", "external_test"])
    p.add_argument("--btsc-block-size", type=int, default=d.btsc_block_size)
    p.add_argument("--gpus", type=int, default=d.gpus)
    p.add_argument("--time-budget-hours", type=float, default=d.time_budget_hours)
    p.add_argument("--resume")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--audit-only", action="store_true", help="hygiene + split + cache, then stop")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate best/ and export")
    p.add_argument("--force-audit", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: first N stems of each source")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end wiring run")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> C.TrainConfig:
    cfg = C.TrainConfig(
        openmed_root=args.openmed_root, btsc_root=args.btsc_root, out_dir=args.out,
        cache_dir=args.cache_dir, zip_dir=args.zip_dir, seed=args.seed, image_size=args.image_size,
        val_frac=args.val_frac, test_frac=args.test_frac, btsc_mode=args.btsc_mode,
        btsc_block_size=args.btsc_block_size, limit_per_source=args.limit,
        variant=args.variant, kernel_size=args.kernel_size, grn=args.grn,
        drop_path=args.drop_path, deep_supervision=not args.no_deep_supervision,
        grad_checkpoint=args.grad_checkpoint, upkern_from=args.upkern_from,
        epochs=args.epochs, batch_size=args.batch, grad_accum=args.grad_accum,
        workers=args.workers, lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, ema_decay=args.ema_decay, patience=args.patience,
        gpus=args.gpus, resume=args.resume, time_budget_hours=args.time_budget_hours,
        amp=not args.no_amp, aug_enabled=not args.no_aug, export_zip=not args.no_export,
        smoke=args.smoke,
    )
    if cfg.smoke:
        cfg.epochs, cfg.max_steps_per_epoch, cfg.patience = 2, 3, 2
        cfg.batch_size, cfg.workers, cfg.variant = 2, 0, "S"
        cfg.split_search_tries = 20
        cfg.pp_thresholds = (0.4, 0.5, 0.6)
        cfg.pp_min_area_fracs = (0.0, 0.001)
        cfg.ema_start_step = 1
    return cfg


def make_specs(cfg: C.TrainConfig) -> tuple[audit_mod.SourceSpec, audit_mod.SourceSpec]:
    om_layout = audit_mod.detect_layout(cfg.openmed_root, "openmed")
    bt_layout = audit_mod.detect_layout(cfg.btsc_root, "btsc")
    openmed = audit_mod.SourceSpec(
        "openmed", cfg.openmed_root, group_of=splits.openmed_group_lookup(cfg.openmed_root),
        group_kind="lesion", group_is_subject_verified=False, **om_layout,
    )
    btsc = audit_mod.SourceSpec(
        "btsc", cfg.btsc_root, group_of=None, group_kind="block",
        group_is_subject_verified=False, **bt_layout,
    )
    return openmed, btsc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-13s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(cfg.out_dir, "logs", "run.log"))],
        force=True,
    )
    started = time.time()

    # ── 1 ───────────────────────────────────────────────────────────────
    logger.info("STAGE 1/7 — environment")
    versions = bootstrap.check_dependencies()
    env = bootstrap.environment_report()
    bootstrap.log_environment(env)
    logger.info("dependencies: %s", versions)
    cfg.gpus = bootstrap.resolve_gpus(cfg.gpus, env, allow_cpu=args.allow_cpu or cfg.smoke or args.audit_only)
    cfg = C.resolve_paths(cfg)
    logger.info("openmed root: %s", cfg.openmed_root)
    logger.info("btsc root   : %s", cfg.btsc_root)
    spec = MedNeXtSpec.from_variant(
        cfg.variant, kernel_size=cfg.kernel_size, grn=cfg.grn, drop_path_rate=cfg.drop_path
    )
    spec_text = spec_summary(spec, count_parameters(build_model(spec, deep_supervision=cfg.deep_supervision)))
    logger.info(spec_text)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    # ── 2 ───────────────────────────────────────────────────────────────
    logger.info("STAGE 2/7 — dataset hygiene")
    openmed_spec, btsc_spec = make_specs(cfg)
    workers = max(2, (os.cpu_count() or 2))
    audits = {
        spec_.name: audit_mod.audit_source(
            spec_, cfg, workers=workers, out_dir=cfg.out_dir, force=args.force_audit
        )
        for spec_ in (openmed_spec, btsc_spec)
    }
    cross_source = audit_mod.cross_source_overlap(audits["openmed"], audits["btsc"], cfg)
    if cross_source["verdict"] != "DISJOINT":
        removed = audit_mod.drop_cross_source_overlap(audits["openmed"], audits["btsc"], cross_source)
        logger.warning("cross-source overlap: removed %d BTSC copies", removed)
    for name, audit in audits.items():
        if audit.checks.failed:
            raise RuntimeError(f"[{name}] hygiene FAIL: " + "; ".join(c.detail for c in audit.checks.failed))
        audit_mod.write_drops(audit.drops, name, os.path.join(cfg.out_dir, "logs", f"dropped_{name}.csv"))

    # ── 3 ───────────────────────────────────────────────────────────────
    logger.info("STAGE 3/7 — splits")
    manifest = splits.build_splits(audits, cfg)
    split_log = audit_mod.CheckLog("split")
    split_report = splits.verify_split(manifest, audits, cfg, split_log)
    if split_log.failed:
        raise RuntimeError("split verification FAIL: " + "; ".join(c.detail for c in split_log.failed))

    # ── 4 ───────────────────────────────────────────────────────────────
    logger.info("STAGE 4/7 — array cache")
    cache_log = audit_mod.CheckLog("cache")
    all_records = manifest.all_records()
    cache = data_mod.build_cache(
        all_records, cfg.resolved_cache_dir(), cfg.image_size, workers, cache_log,
        seed=cfg.seed, force=args.rebuild_cache,
    )
    vanished = data_mod.vanished_records(cache, all_records)
    if vanished:
        logger.warning("dropping %d records whose mask vanishes at %dpx", len(vanished), cfg.image_size)
        data_mod.drop_vanished(manifest, vanished)
    manifest.save(os.path.join(cfg.out_dir, "splits"))
    for split in ("train", "val", "test"):
        logger.info("  %-5s %d slices %s", split, len(manifest.part(split)),
                    {s: sum(1 for r in manifest.part(split) if r.source == s) for s in ("openmed", "btsc")})
    if not manifest.train or not manifest.val:
        raise RuntimeError("Empty train or validation split — cannot train.")

    hygiene = {
        "sources": {n: a.to_dict() for n, a in audits.items()},
        "cross_source": {k: v for k, v in cross_source.items() if k != "shared_pairs"},
        "split": split_log.to_list(), "split_report": split_report, "cache": cache_log.to_list(),
    }
    with open(os.path.join(cfg.out_dir, "logs", "hygiene.json"), "w") as handle:
        json.dump(plain(hygiene), handle, indent=1)

    final_log = audit_mod.CheckLog("final")
    results: dict[str, Any] = {}

    if not args.audit_only:
        # ── 5 ───────────────────────────────────────────────────────────
        if args.eval_only:
            logger.info("STAGE 5/7 — skipped (--eval-only)")
        else:
            logger.info("STAGE 5/7 — training on %d device(s)", max(1, cfg.gpus))
            world = max(1, cfg.gpus)
            if world > 1:
                import torch.multiprocessing as mp

                mp.spawn(engine.run_worker, nprocs=world, args=(world, cfg.to_dict()), join=True)
            else:
                engine.run_worker(0, 1, cfg.to_dict())

        # ── 6 ───────────────────────────────────────────────────────────
        logger.info("STAGE 6/7 — post-processing search, held-out test, export")
        results = evaluate.finalise(cfg, manifest, cache, final_log)

        # ── 7 ───────────────────────────────────────────────────────────
        logger.info("STAGE 7/7 — plots, report, ZIP")
        history = reporting._load_json(os.path.join(cfg.out_dir, "metrics", "history.json"), [])
        reporting.plot_learning_curves(history, cfg.out_dir)
        reporting.plot_dataset_overview(manifest, audits, cfg.out_dir)
        reporting.plot_btsc_adjacency(audits, manifest, cfg.out_dir, cfg.seed)
        reporting.plot_samples(cache, manifest, cfg, cfg.out_dir)
        reporting.plot_test_distribution(results["test_rows"], results["summary"]["test"] or {"by_size": {}}, cfg.out_dir)
        reporting.plot_postprocess_grid(results["tuning"], cfg, cfg.out_dir)
        if results["summary"]["test"]:
            reporting.plot_calibration(cfg.out_dir, results["summary"]["test"])
        reporting.plot_qualitative(cfg, results["test_rows"], manifest, cfg.out_dir)

    report_path = reporting.write_report(
        cfg, cfg.out_dir, env=env, audits=audits, cross_source=cross_source,
        split_log=split_log.to_list() + final_log.to_list(), split_report=split_report,
        cache_log=cache_log.to_list(), spec_text=spec_text,
        summary=results.get("summary"),
    )
    logger.info("report: %s", report_path)

    if cfg.export_zip and not args.audit_only:
        logger.info("ZIP: %s", reporting.export_zip(cfg.out_dir, cfg.zip_dir))

    logger.info("DONE in %.1f min. Artefacts under %s", (time.time() - started) / 60, cfg.out_dir)
    if results.get("summary", {}).get("test"):
        test = results["summary"]["test"]["overall"]
        logger.info("FINAL — test Dice %.4f (median %.4f) | IoU %.4f | HD95 median %s px",
                    test["dice"]["mean"], test["dice"]["median"], test["iou"]["mean"],
                    f"{test['hd95_px']['median']:.1f}" if test["hd95_px"]["median"] is not None else "n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
