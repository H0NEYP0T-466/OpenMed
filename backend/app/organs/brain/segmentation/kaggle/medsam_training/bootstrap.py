"""Stage 1 — environment, dependencies and pretrained weights.

Runs before anything else so a Kaggle session fails fast and legibly:

  1. Log the environment (torch, CUDA, GPU names, VRAM, disk).
  2. Verify every required dependency imports.
  3. Ensure ``lite_medsam.pth`` is present and *provably* loadable — build the
     architecture and apply the state dict with ``strict=True``. A bad
     download fails here, not at epoch 1.

The weights are fetched from the same official Google Drive file the backend
uses (``prepare_weights.py``), so training and serving start from identical
initialisation. The download is staged to ``.part`` and only promoted after
verification, so a failed download can never clobber a good checkpoint.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import sys
import time
from typing import Any, Optional

from . import config as C

logger = logging.getLogger("bootstrap")

REQUIRED_PACKAGES = (
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("timm", "timm"),
    ("gdown", "gdown"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> dict[str, str]:
    """Import every required package. Returns ``{module: version}``.

    Raises
    ------
    RuntimeError
        Listing every missing package at once, with the pip name to install.
    """
    import importlib

    versions: dict[str, str] = {}
    missing: list[str] = []

    for module, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - environment dependent
            missing.append(f"{module} (pip install {pip_name}) — {exc}")
            continue
        versions[module] = str(getattr(mod, "__version__", "unknown"))

    if missing:
        raise RuntimeError(
            "Missing required dependencies:\n  - " + "\n  - ".join(missing) +
            "\nOn Kaggle most of these are preinstalled; only 'gdown' and 'timm' "
            "may need `!pip install -q gdown timm`."
        )
    return versions


def environment_report() -> dict[str, Any]:
    """Collect everything worth logging about the runtime."""
    import torch

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "disk": {},
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                    "bf16": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
                }
            )

    for path in ("/kaggle/working", "/kaggle/input", "/tmp"):
        if os.path.isdir(path):
            try:
                usage = shutil.disk_usage(path)
                report["disk"][path] = {
                    "total_gb": round(usage.total / 1024**3, 2),
                    "free_gb": round(usage.free / 1024**3, 2),
                }
            except OSError:
                pass

    return report


def log_environment(report: dict[str, Any]) -> None:
    logger.info("=" * 72)
    logger.info("ENVIRONMENT")
    logger.info("=" * 72)
    logger.info("python      : %s", report["python"])
    logger.info("platform    : %s", report["platform"])
    logger.info("torch       : %s (cuda %s)", report["torch"], report["cuda_version"])
    logger.info("cuda avail  : %s   gpu_count=%s", report["cuda_available"], report["gpu_count"])
    for gpu in report["gpus"]:
        logger.info(
            "  gpu %d     : %s  %.1f GB  sm_%s  bf16=%s",
            gpu["index"], gpu["name"], gpu["vram_gb"], gpu["capability"], gpu["bf16"],
        )
    for path, usage in report["disk"].items():
        logger.info("disk %-14s free %.1f / %.1f GB", path, usage["free_gb"], usage["total_gb"])
    logger.info("=" * 72)


def assert_training_hardware(report: dict[str, Any], expected_gpus: int) -> None:
    """Fail loudly when the run cannot use the hardware it was configured for.

    A silent fallback to CPU on Kaggle means a 100-epoch run that never
    finishes inside the 12 h session limit. Better to stop at second 5.
    """
    if expected_gpus > 1 and report["gpu_count"] < expected_gpus:
        raise RuntimeError(
            f"Configured for {expected_gpus} GPUs but only {report['gpu_count']} "
            "are visible. On Kaggle: Settings -> Accelerator -> GPU T4 x2. "
            "Pass --gpus 1 to run single-GPU intentionally."
        )
    if expected_gpus >= 1 and not report["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available. On Kaggle: Settings -> Accelerator -> GPU T4 x2."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained weights
# ─────────────────────────────────────────────────────────────────────────────

def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _looks_valid(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) >= C.EXPECTED_MIN_BYTES


def download_weights(dest: str, file_id: Optional[str] = None, force: bool = False) -> str:
    """Fetch ``lite_medsam.pth`` from Google Drive into ``dest``.

    Downloads to ``dest.part`` first. Returns the path to the verified-size
    file (which may be ``dest`` itself when it already existed).
    """
    fid = file_id or os.getenv("OPENMED_SEG_GDRIVE_ID", C.GDRIVE_FILE_ID)

    if _looks_valid(dest) and not force:
        logger.info("Checkpoint already present: %s (%.1f MB)",
                    dest, os.path.getsize(dest) / 1024**2)
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(part):
        os.remove(part)

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("gdown is required to fetch the weights: pip install gdown") from exc

    url = f"https://drive.google.com/uc?id={fid}"
    logger.info("Downloading LiteMedSAM weights from Google Drive...")
    started = time.time()
    out = gdown.download(url, part, quiet=False)

    if out is None or not os.path.isfile(part):
        raise RuntimeError(
            f"Download failed. Check connectivity, or the Drive file ID ({fid}). "
            f"Manual fallback: {url}"
        )

    size = os.path.getsize(part)
    if size < C.EXPECTED_MIN_BYTES:
        os.remove(part)
        raise RuntimeError(
            f"Downloaded file is {size} bytes, expected >= {C.EXPECTED_MIN_BYTES}. "
            "Google Drive often returns an HTML quota page instead of the file. "
            f"Download manually and upload as a Kaggle dataset: {url}"
        )

    logger.info("Downloaded %.1f MB in %.1f s", size / 1024**2, time.time() - started)
    return part


def verify_weights(path: str, device: str = "cpu") -> dict[str, Any]:
    """Build the architecture, apply the state dict strictly, run one forward.

    This is the authoritative check — a size gate alone cannot tell a real
    checkpoint from a truncated one.
    """
    import torch

    from .medsam_model import build_medsam_lite

    if not _looks_valid(path):
        raise RuntimeError(f"Checkpoint missing or too small: {path}")

    logger.info("Building LiteMedSAM architecture...")
    model = build_medsam_lite()
    n_params = sum(p.numel() for p in model.parameters())

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(
            f"{path} is not a state dict (got {type(state).__name__}). "
            "Expected torch.save(model.state_dict(), ...)."
        )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with the LiteMedSAM architecture: {exc}"
        ) from exc
    logger.info("State dict applied with strict=True (%d tensors).", len(state))

    # The Attention module caches a bias table in train() and drops it in
    # eval(). Call both so the cache exists before any inference.
    model.eval()
    model.to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, device=device)
        box = torch.tensor([[[60.0, 70.0, 190.0, 210.0]]], device=device)
        masks, iou = model(dummy, boxes=box)

    if tuple(masks.shape) != (1, 1, C.IMAGE_SIZE, C.IMAGE_SIZE):
        raise RuntimeError(
            f"Forward pass returned {tuple(masks.shape)}, expected "
            f"(1, 1, {C.IMAGE_SIZE}, {C.IMAGE_SIZE})."
        )

    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "sha256": sha256_of(path),
        "parameters": n_params,
        "parameters_millions": round(n_params / 1e6, 2),
        "forward_ok": True,
        "forward_mask_shape": list(masks.shape),
        "forward_iou_shape": list(iou.shape),
    }


def prepare_pretrained(cfg: C.TrainConfig, force: bool = False) -> tuple[str, dict[str, Any]]:
    """Ensure a verified pretrained checkpoint exists; return ``(path, info)``.

    Resolution order:
      1. ``cfg.pretrained`` if the caller pinned one (e.g. a Kaggle dataset
         holding ``lite_medsam.pth`` — avoids the Drive download entirely).
      2. Any ``lite_medsam.pth`` already under ``/kaggle/input``.
      3. Download into ``cfg.weights_dir``.
    """
    import glob

    dest = os.path.join(cfg.weights_dir, C.WEIGHT_FILENAME)

    candidate: Optional[str] = None
    if cfg.pretrained and os.path.isfile(cfg.pretrained):
        candidate = cfg.pretrained
        logger.info("Using caller-supplied checkpoint: %s", candidate)
    else:
        hits = sorted(glob.glob("/kaggle/input/**/lite_medsam.pth", recursive=True))
        if hits:
            candidate = hits[0]
            logger.info("Found pretrained checkpoint on a mounted dataset: %s", candidate)

    if candidate and _looks_valid(candidate):
        info = verify_weights(candidate)
        info["source"] = "mounted"
        cfg.pretrained = candidate
        return candidate, info

    downloaded = download_weights(dest, force=force)
    info = verify_weights(downloaded)
    info["source"] = "google_drive"

    if downloaded != dest:
        shutil.move(downloaded, dest)
        info["path"] = dest
    cfg.pretrained = dest
    return dest, info
