#!/usr/bin/env python
"""OpenMed — lungs segmentation. Single Kaggle entry point.

    !git clone <repo> && cd <repo>
    !python backend/app/organs/lungs/segmentation/kaggle/train_lungs_seg.py

Stages
------
  1   environment, dependency check, GPU assertion, encoder round-trip self-check
  2   dataset discovery (bounded walk, never a hardcoded path) and catalog build
  3   hygiene over every image: pairing, geometry, channels, mask values, lung
      extent and morphology, infection extent, folder agreement, exact and
      perceptual duplicates, BIDS subject groups
  4   leak-free grouped split (clusters and subjects, never images) + manifest
  5   uint8 array cache built through the shared preprocessing contract, verified
      against the serving decode
  6   optional regularisation probe: short runs on a fraction of train, pick
      augmentation strength on validation, then throw the weights away
  7   DDP training: AdamW, fp16 AMP, EMA, augmentation, early stopping, budget
  8   post-processing search on validation, one pass over test through the
      serving pipeline, final self-describing checkpoint
  9   nine figures, TRAINING_REPORT.txt, and a ZIP in /kaggle/working

Nothing is deleted from the dataset. Hygiene only *records* drops; every
exclusion is a row in ``dropped_files.csv`` and the cache simply skips it. The
dataset is read-only, at runtime, wherever Kaggle mounts it.

Expected mount (auto-detected, override with ``--data-root``):
    /kaggle/input/<dataset>/Lung Segmentation Data/Lung Segmentation Data/{Train,Val,Test}/...
    /kaggle/input/<dataset>/Infection Segmentation Data/Infection Segmentation Data/{...}

Deployment: copy ``best/lungs_seg.pth`` and ``best/lungs_seg.sidecar.json`` to
``backend/app/organs/lungs/segmentation/checkpoints/``. Nothing else.

    !python .../train_lungs_seg.py --smoke                 # wiring check, minutes, CPU ok
    !python .../train_lungs_seg.py --audit-only           # data report, no training
    !python .../train_lungs_seg.py                        # stage lung, 100 epochs
    !python .../train_lungs_seg.py --stage twohead --init-from best/lungs_seg.pth
    !python .../train_lungs_seg.py --resume auto          # continue a killed run
    !python .../train_lungs_seg.py --eval-only            # re-tune + re-test an existing best
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from typing import Any


# The package uses relative imports (``from ..checkpoint import ...``), so it has to be
# imported as ``app.organs.lungs.segmentation.<module>``. Find the backend root by
# walking up for the directory that holds ``app/`` and ``tests/``, then put it on the
# path. On Kaggle the repo is cloned with the same layout, so this holds there too.
def _backend_root(start: str) -> str:
    here = os.path.dirname(os.path.abspath(start))
    while True:
        if os.path.isdir(os.path.join(here, "app")) and os.path.isdir(os.path.join(here, "tests")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError(
                f"Could not find the backend root (a directory holding app/ and tests/) above "
                f"{os.path.abspath(start)}. Run this from a clone of the OpenMed repository."
            )
        here = parent


_BACKEND = _backend_root(__file__)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch  # noqa: E402

from app.organs.lungs.segmentation import checkpoint as ckpt  # noqa: E402
from app.organs.lungs.segmentation.kaggle import (  # noqa: E402
    bootstrap,  # noqa: E402
    engine,
    evaluate,
    reporting,
    splits,
)
from app.organs.lungs.segmentation.kaggle import catalog as catalog_mod  # noqa: E402
from app.organs.lungs.segmentation.kaggle import config as C  # noqa: E402
from app.organs.lungs.segmentation.kaggle import data as data_mod  # noqa: E402
from app.organs.lungs.segmentation.kaggle import hygiene as hygiene_mod  # noqa: E402
from app.organs.lungs.segmentation.labels import TASK_NAME  # noqa: E402

logger = logging.getLogger("train_lungs_seg")

MANIFEST = "split_manifest.csv"
HYGIENE_JSON = "hygiene.json"
SPLIT_JSON = "split_report.json"
CONFIG_JSON = "config.json"
HISTORY_JSON = "history.json"
DONE_JSON = "training_done.json"
PROBE_JSON = "probe_results.json"


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = C.TrainConfig()
    p = argparse.ArgumentParser(
        description="Train the OpenMed lungs segmenter (lung field + COVID opacity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", help="directory holding both COVID-QU-Ex trees")
    p.add_argument("--search-root", action="append", default=[],
                   help="root to walk when --data-root is not given (default /kaggle/input)")
    p.add_argument("--out", default="", help="run directory")
    p.add_argument("--cache-dir", help="uint8 cache directory (default /kaggle/temp/...)")
    p.add_argument("--zip-dir", default=d.zip_dir, help="where the export ZIP is written")
    p.add_argument("--stage", default=d.stage, choices=list(C.STAGES))
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch", type=int, default=d.batch_size, help="per GPU")
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--encoder-lr-mult", type=float, default=d.encoder_lr_mult)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--ema-decay", type=float, default=d.ema_decay)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--gpus", type=int, default=d.gpus)
    p.add_argument("--time-budget-hours", type=float, default=d.time_budget_hours)
    p.add_argument("--pretrained", default=d.pretrained,
                   help="HF hub id or local directory for the MiT-B0 encoder")
    p.add_argument("--allow-random-init", action="store_true",
                   help="train the encoder from scratch instead of failing loudly")
    p.add_argument("--init-from", help="serving checkpoint for a WEIGHTS-ONLY start")
    p.add_argument("--split-from", help="reuse another run's split_manifest.csv")
    p.add_argument("--resume", default=d.resume,
                   help="auto | none | path to a last.pt")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-tta", action="store_true", help="disable flip TTA in the search")
    p.add_argument("--no-export", action="store_true", help="skip the ZIP")
    p.add_argument("--continue-on-fail", action="store_true",
                   help="drop hygiene FAIL offenders instead of aborting")
    p.add_argument("--group-by-subject", default=None, action="store_true",
                   help="force BIDS subject grouping (on by default)")
    p.add_argument("--no-group-by-subject", dest="group_by_subject", action="store_false")
    p.add_argument("--probe", action="store_true",
                   help="run the short augmentation probe before the real run")
    p.add_argument("--probe-epochs", type=int, default=d.probe_epochs)
    p.add_argument("--limit", type=int, default=0, help="debug: first N pairs per class/split")
    p.add_argument("--audit-only", action="store_true", help="hygiene + split, then stop")
    p.add_argument("--eval-only", action="store_true", help="skip training; re-tune and re-test")
    p.add_argument("--force-audit", action="store_true", help="rerun hygiene even if a manifest exists")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="permit a CPU run (wiring only)")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end wiring run")
    p.add_argument("--force-resume", action="store_true",
                   help="resume even when the config fingerprint changed")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> C.TrainConfig:
    cfg = C.TrainConfig(
        data_root=args.data_root, out_dir=args.out, cache_dir=args.cache_dir,
        zip_dir=args.zip_dir, split_from=args.split_from, init_from=args.init_from,
        pretrained=args.pretrained, allow_random_init=args.allow_random_init,
        stage=args.stage, seed=args.seed, split_fracs=C.TrainConfig().split_fracs,
        limit_per_class=args.limit,
        epochs=args.epochs, batch_size=args.batch, grad_accum=args.grad_accum,
        workers=args.workers, lr=args.lr, encoder_lr_mult=args.encoder_lr_mult,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        ema_decay=args.ema_decay, patience=args.patience, gpus=args.gpus,
        resume=args.resume, force_resume=args.force_resume,
        time_budget_hours=args.time_budget_hours, amp=not args.no_amp,
        aug_enabled=not args.no_aug, tta_hflip=not args.no_tta,
        export_zip=not args.no_export, continue_on_fail=args.continue_on_fail,
        probe=bool(args.probe), probe_epochs=args.probe_epochs, smoke=args.smoke,
    )
    if args.group_by_subject is not None:
        cfg.group_by_subject = bool(args.group_by_subject)
    if cfg.smoke:
        # A smoke run shortens the schedule, but never overrides a value the caller set on
        # purpose: --smoke --epochs 4 is how a resume is tested without a GPU.
        defaults = C.TrainConfig()
        cfg.max_steps_per_epoch = 3
        if args.epochs == defaults.epochs:
            cfg.epochs = 2
        if args.patience == defaults.patience:
            cfg.patience = 2
        cfg.batch_size, cfg.workers, cfg.probe_epochs = 2, 0, 1
        cfg.encoder, cfg.allow_random_init = "debug-tiny", True
        cfg.pp_thresholds, cfg.pp_lung_min_area_fracs = (0.4, 0.5), (0.0, 0.001)
        cfg.pp_opacity_min_area_fracs = (0.0, 0.001)
        cfg.pp_tune_max_images, cfg.ema_start_step = 40, 1
        cfg.probe = False
    return cfg


# ---------------------------------------------------------------------- plumbing


def configure_logging(out_dir: str) -> None:
    """stdout for the notebook cell, and a file that survives a dead kernel."""
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s  %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    path = os.path.join(out_dir, "logs", "train_lungs_seg.log")
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    for noisy in ("PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.info("log file: %s", path)


def read_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def banner(step: int, total: int, title: str) -> None:
    logger.info("")
    logger.info("── stage %d/%d  %s %s", step, total, title, "─" * max(0, 46 - len(title)))


def prune_empty_audit_dirs(path: str) -> None:
    """A stage that failed may have created an empty folder. Leave nothing half-made."""
    for name in ("metrics", "plots"):
        folder = os.path.join(path, name)
        if os.path.isdir(folder) and not os.listdir(folder):
            os.rmdir(folder)


# ----------------------------------------------------------------------- stages


def stage_environment(cfg: C.TrainConfig, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    env = bootstrap.environment_report()
    bootstrap.log_environment(env)
    need_transformers = cfg.encoder != "debug-tiny"
    versions = bootstrap.check_dependencies(need_transformers=need_transformers)
    logger.info("dependencies: %s", ", ".join(f"{k} {v}" for k, v in sorted(versions.items())))
    world = bootstrap.resolve_gpus(cfg.gpus, env, allow_cpu=args.allow_cpu)
    if need_transformers and env["gpu_count"]:
        report = bootstrap.encoder_selfcheck()
        logger.info("encoder self-check: %s", json.dumps(report)[:240])
    return env, world


def stage_data(cfg: C.TrainConfig, args: argparse.Namespace) -> dict[str, str]:
    searches = tuple(args.search_root) if args.search_root else (C.KAGGLE_INPUT,)
    roots = C.resolve_data_roots(cfg, search_roots=searches)
    for kind, path in sorted(roots.items()):
        logger.info("%s tree: %s", kind, path)
    return roots


def stage_audit(
    cfg: C.TrainConfig, roots: dict[str, str]
) -> tuple[splits.SplitResult, dict[str, Any], dict[str, Any]]:
    catalog = catalog_mod.build_catalog(roots, cfg)
    logger.info(
        "catalog: %d records (%d lung-tree, %d infection-tree, %d opacity-supervised)",
        len(catalog.records), sum(1 for r in catalog.records if not r.in_infection_tree)
        if catalog.records else 0,
        sum(1 for r in catalog.records if r.in_infection_tree),
        sum(1 for r in catalog.records if r.opacity_supervised),
    )
    hygiene = hygiene_mod.run_hygiene(catalog, cfg, cfg.out_dir, workers=cfg.workers)
    logger.info(
        "hygiene: kept %d of %d (%d dropped); clusters %s",
        len(hygiene.records), len(hygiene.all_records),
        len(hygiene.all_records) - len(hygiene.records),
        hygiene.clusters.stats.get("clusters", "?"),
    )

    split = splits.build_split(
        hygiene.records, hygiene.all_records, hygiene.clusters, hygiene.cluster_scope, cfg
    )
    splits.write_manifest(split, os.path.join(cfg.out_dir, MANIFEST))
    splits.write_report(split.report, os.path.join(cfg.out_dir, SPLIT_JSON))
    logger.info(
        "split: train %d / val %d / test %d | moved %s images against the shipped split",
        len(split.part("train")), len(split.part("val")), len(split.part("test")),
        f"{split.report.get('images_moved', 0):,}",
    )
    return split, hygiene.report, split.report


def stage_cache(
    cfg: C.TrainConfig, split: splits.SplitResult, force: bool
) -> tuple[data_mod.ArrayCache, Any]:
    records = [record for name in C.SPLITS for record in split.part(name)]
    cache = data_mod.build_cache(
        records, cfg.resolved_cache_dir(), workers=cfg.workers, force=force
    )
    log = hygiene_mod.CheckLog("cache")
    data_mod.verify_cache(cache, records, log, n=96, seed=cfg.seed)
    logger.info("cache ready: %d images at %s", len(records), cfg.resolved_cache_dir())
    return cache, log


def stage_probe(cfg: C.TrainConfig, world: int) -> float | None:
    """Short runs, one augmentation strength at a time, judged on validation.

    Cheap by construction: ``is_probe`` skips the best/last checkpoints, so nothing here
    can be mistaken for a real result. The winner is only a number to set on the real run.
    """
    strengths = (0.6, 1.0, 1.4)
    results: list[dict[str, Any]] = []
    manifest = os.path.join(cfg.out_dir, MANIFEST)
    for strength in strengths:
        out = os.path.join(cfg.out_dir, f"probe_aug{strength:.1f}")
        probe_cfg = C.TrainConfig.from_dict({
            **cfg.to_dict(), "aug_strength": strength, "epochs": cfg.probe_epochs,
            "is_probe": True, "out_dir": out, "resume": "none", "export_zip": False,
            "probe": False, "stage": cfg.stage,
        })
        os.makedirs(out, exist_ok=True)
        shutil.copy2(manifest, os.path.join(out, MANIFEST))
        probe_cfg.save(os.path.join(out, CONFIG_JSON))
        logger.info("probe: aug_strength %.1f, %d epoch(s) on %.0f%% of train",
                    strength, probe_cfg.epochs, probe_cfg.probe_train_frac * 100)
        try:
            train(probe_cfg, world)
            history = read_json(os.path.join(out, HISTORY_JSON)) or []
            score = max((row["best_score"] for row in history), default=None)
        except Exception as exc:  # noqa: BLE001 - a probe must never kill the real run
            logger.warning("probe at aug_strength %.1f failed: %s", strength, exc)
            score = None
        results.append({"aug_strength": strength, "score": score})
        logger.info("probe: aug_strength %.1f -> %s", strength,
                    "n/a" if score is None else f"{score:.4f}")

    usable = [row for row in results if row["score"] is not None]
    winner = max(usable, key=lambda row: row["score"])["aug_strength"] if usable else None
    with open(os.path.join(cfg.out_dir, PROBE_JSON), "w", encoding="utf-8") as handle:
        json.dump({"results": results, "chosen_aug_strength": winner}, handle, indent=1)
    logger.info("probe: chosen augmentation strength %s (results in %s)",
                winner, PROBE_JSON)
    return winner


def train(cfg: C.TrainConfig, world: int) -> None:
    payload = cfg.to_dict()
    if world > 1:
        torch.multiprocessing.spawn(engine.run_worker, args=(world, payload), nprocs=world,
                                    join=True)
    else:
        engine.run_worker(0, 1, payload)


def stage_finalise(
    cfg: C.TrainConfig, split: splits.SplitResult, cache: data_mod.ArrayCache, log: Any
) -> dict[str, Any]:
    best = os.path.join(cfg.out_dir, "best", ckpt.CHECKPOINT_NAME)
    logger.info("finalising from %s", best)
    return evaluate.finalise(cfg, split, cache, log, best, cfg.heads())


def stage_figures(
    cfg: C.TrainConfig,
    split: splits.SplitResult,
    cache: data_mod.ArrayCache | None,
    hygiene_report: dict[str, Any],
    split_report: dict[str, Any],
    summary: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> None:
    """Every figure is optional: one that fails is logged and skipped, never fatal."""
    reporting.plot_dataset_overview(split_report, hygiene_report, cfg.out_dir)
    reporting.plot_duplicate_clusters(split_report, cfg.out_dir)
    if cache is None:
        return
    history = read_json(os.path.join(cfg.out_dir, HISTORY_JSON)) or []
    reporting.plot_learning_curves({cfg.stage: history}, cfg.out_dir)
    reporting.plot_samples(cache, split, cfg, cfg.out_dir)
    reporting.plot_augmentation_preview(cache, split, cfg, cfg.out_dir)
    if not summary or not final:
        return
    reporting.plot_test_distributions(summary.get("test") or {}, final.get("test_rows") or [],
                                      cfg.out_dir)
    # the grids are not in summary.json (they are large); finalise writes them separately
    grids = read_json(os.path.join(cfg.out_dir, "metrics", "postprocess_grid.json")) or {}
    reporting.plot_postprocess_grid(grids, cfg.out_dir)
    reporting.plot_calibration(summary.get("test") or {}, cfg.out_dir)
    reporting.plot_qualitative(
        cache, split, final.get("test_rows") or [], cfg.out_dir,
        os.path.join(cfg.out_dir, "best", ckpt.CHECKPOINT_NAME),
    )


def publish_best(cfg: C.TrainConfig) -> str | None:
    """Put the deliverable next to the ZIP so it is one click away on Kaggle."""
    best = os.path.join(cfg.out_dir, "best", ckpt.CHECKPOINT_NAME)
    if not os.path.isfile(best):
        return None
    os.makedirs(cfg.zip_dir, exist_ok=True)
    for path in (best, ckpt.sidecar_path_for(best)):
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(cfg.zip_dir, os.path.basename(path)))
    return os.path.join(cfg.zip_dir, ckpt.CHECKPOINT_NAME)


def log_headline(summary: dict[str, Any] | None) -> None:
    """The three numbers the task is defined by, never one."""
    test = (summary or {}).get("test") or {}
    lung = test.get("lung_field") or {}
    opacity = test.get("opacity")
    logger.info("")
    logger.info("=" * 72)
    if lung:
        hd95 = lung["hd95_px"]["median"]
        logger.info("LUNG FIELD   Dice %.4f mean / %.4f median | IoU %.4f | HD95 %s px median",
                    lung["dice"]["mean"], lung["dice"]["median"], lung["iou"]["mean"],
                    "nan" if hd95 is None else f"{hd95:.1f}")
    if opacity:
        positives, normal, non_covid = opacity["positives"], opacity["normal"], opacity["non_covid"]
        logger.info("OPACITY      positive Dice %.4f mean / %.4f median on %d COVID-19 images",
                    positives["dice"]["mean"], positives["dice"]["median"], positives["n"])
        logger.info("NORMAL FPR   %.4f (%d of %d Normal images predicted opacity)",
                    normal["false_positive_rate"], normal["n_with_prediction"], normal["n"])
        logger.info("NON-COVID    any opacity %.4f, mean fraction %.4f (behaviour, not a score)",
                    non_covid["any_opacity_rate"], non_covid["mean_predicted_fraction_of_lung"])
    else:
        logger.info("OPACITY      head not trained in this run (stage lung)")
    logger.info("=" * 72)


# ------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    out_dir = cfg.resolved_out_dir()
    cfg.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    configure_logging(out_dir)
    cfg.save(os.path.join(out_dir, CONFIG_JSON))
    if cfg.encoder == ckpt.ENCODER_DEBUG:
        # The serving loader refuses debug-encoder checkpoints on purpose. A smoke run has to
        # exercise the same finalise path, so it opts in explicitly and says so.
        os.environ[ckpt.ALLOW_DEBUG_ENV] = "1"
        logger.warning(
            "debug encoder: %s=1 set for this process only. The checkpoint this run produces is a "
            "wiring artefact and must never be served or reported as a result.",
            ckpt.ALLOW_DEBUG_ENV,
        )

    started = time.time()
    logger.info("=" * 72)
    logger.info("OpenMed — %s", TASK_NAME)
    logger.info("stage %s | heads %s | seed %d | run directory %s",
                cfg.stage, list(cfg.heads()), cfg.seed, out_dir)
    logger.info("=" * 72)
    logger.info("nothing in the dataset is modified: hygiene records drops, the cache skips them")

    total_steps = 9
    banner(1, total_steps, "environment")
    env, world = stage_environment(cfg, args)
    cfg.gpus = world

    manifest_path = os.path.join(out_dir, MANIFEST)
    resume_manifest = (
        not args.force_audit
        and os.path.isfile(manifest_path)
        and read_json(os.path.join(out_dir, HYGIENE_JSON)) is not None
    )

    split: splits.SplitResult
    hygiene_report: dict[str, Any]
    split_report: dict[str, Any]
    if resume_manifest:
        banner(3, total_steps, "audit (reused from disk)")
        split = splits.load_manifest(manifest_path)
        hygiene_report = read_json(os.path.join(out_dir, HYGIENE_JSON)) or {}
        split_report = read_json(os.path.join(out_dir, SPLIT_JSON)) or {}
        logger.info(
            "SPLIT REUSED from %s | train %d / val %d / test %d | hygiene from %s",
            MANIFEST, len(split.part("train")), len(split.part("val")), len(split.part("test")),
            HYGIENE_JSON,
        )
        logger.info("pass --force-audit to recompute hygiene and the split from scratch")
    else:
        banner(2, total_steps, "dataset discovery")
        roots = stage_data(cfg, args)
        banner(3, total_steps, "hygiene and grouped split")
        split, hygiene_report, split_report = stage_audit(cfg, roots)

    if args.audit_only:
        banner(9, total_steps, "figures and report (no training)")
        stage_figures(cfg, split, None, hygiene_report, split_report, None, None)
        path = reporting.write_report(
            cfg, out_dir, env=env, hygiene=hygiene_report, split_report=split_report,
            stages={}, summary=None, history={},
        )
        logger.info("audit-only: %s", path)
        prune_empty_audit_dirs(out_dir)
        logger.info("audit-only finished in %.1f min", (time.time() - started) / 60)
        return 0

    banner(5, total_steps, "array cache")
    cache, cache_log = stage_cache(cfg, split, force=args.rebuild_cache)
    merged = list(hygiene_report.get("checks") or []) + cache_log.to_list()
    hygiene_report["checks"] = merged
    hygiene_report["tally"] = _tally(merged)
    hygiene_mod.write_hygiene_json(hygiene_report, os.path.join(out_dir, HYGIENE_JSON))

    if cfg.probe and not args.eval_only:
        banner(6, total_steps, "augmentation probe")
        chosen = stage_probe(cfg, world)
        if chosen is not None:
            cfg.aug_strength = chosen
            cfg.save(os.path.join(out_dir, CONFIG_JSON))

    if not args.eval_only:
        banner(7, total_steps, "training")
        train(cfg, world)

    banner(8, total_steps, "post-processing search and held-out test")
    finalise_log = hygiene_mod.CheckLog("final")
    final = stage_finalise(cfg, split, cache, finalise_log)
    merged = list(hygiene_report.get("checks") or []) + finalise_log.to_list()
    hygiene_report["checks"] = merged
    hygiene_report["tally"] = _tally(merged)
    hygiene_mod.write_hygiene_json(hygiene_report, os.path.join(out_dir, HYGIENE_JSON))

    summary = final["summary"]
    banner(9, total_steps, "figures, report, zip")
    stage_figures(cfg, split, cache, hygiene_report, split_report, summary, final)
    done = read_json(os.path.join(out_dir, DONE_JSON)) or {}
    history = read_json(os.path.join(out_dir, HISTORY_JSON)) or []
    report_path = reporting.write_report(
        cfg, out_dir, env=env, hygiene=hygiene_report, split_report=split_report,
        stages={cfg.stage: done}, summary=summary, history={cfg.stage: history},
    )
    logger.info("report: %s", report_path)
    log_headline(summary)

    published = publish_best(cfg)
    if published:
        logger.info("checkpoint ready to deploy: %s (+ sidecar)", published)
    if cfg.export_zip:
        zip_path = reporting.export_zip(out_dir, cfg.zip_dir)
        logger.info("zip: %s (%.1f MB)", zip_path, os.path.getsize(zip_path) / 2**20)
    logger.info("finished in %.2f h", (time.time() - started) / 3600)
    return 0


def _tally(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
