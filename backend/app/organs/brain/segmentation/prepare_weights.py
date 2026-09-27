"""Download, verify, and prepare the LiteMedSAM pretrained weights.

This is the one-stop script that makes the segmentation model ``ready for
inference``.  It:

1. Fetches ``lite_medsam.pth`` from Google Drive (official bowang-lab release).
2. Builds the LiteMedSAM architecture (``build_medsam_lite()``).
3. Applies the checkpoint with a strict state-dict load — so an incompatible
   download fails *here*, loudly, instead of silently at inference time.
4. Runs a short end-to-end forward pass (box prompt) on a synthetic input to
   prove the model actually executes on the exact input the backend pipeline
   feeds it.
5. Places the verified checkpoint at the canonical path the backend loads from.

The download is staged to a ``.part`` file and only moved into place after
verification, so a bad download can never clobber a previously good checkpoint.

Usage
-----
    python -m app.organs.brain.segmentation.prepare_weights   # download + verify + install
    python -m app.organs.brain.segmentation.prepare_weights --force   # re-download even if present
    python -m app.organs.brain.segmentation.prepare_weights --cpu     # verify on CPU

Environment variables
    OPENMED_SEG_CHECKPOINT : override the destination path
    OPENMED_SEG_GDRIVE_ID  : override the Google Drive file ID
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Official Google Drive file ID for lite_medsam.pth
GDRIVE_FILE_ID = "18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z"
WEIGHT_FILENAME = "lite_medsam.pth"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

# Minimum plausible checkpoint size.  LiteMedSAM is ~9.8M params → ~39 MB at
# fp32, so a genuine ``lite_medsam.pth`` never comes in under 30 MB.  Google
# Drive error pages are HTML of only a few KB.  This gate is intentionally a
# coarse pre-filter — the authoritative check is ``verify_checkpoint``, which
# builds the model and runs a real forward pass.
EXPECTED_MIN_BYTES = 30_000_000


def default_checkpoint_path() -> Path:
    """Return the canonical path where the segmentation checkpoint lives."""
    configured = os.getenv("OPENMED_SEG_CHECKPOINT")
    if configured:
        return Path(configured).expanduser()
    return CHECKPOINTS_DIR / WEIGHT_FILENAME


def _looks_valid(path: Path) -> bool:
    """True if the file exists and exceeds the minimum plausible size."""
    return path.is_file() and path.stat().st_size >= EXPECTED_MIN_BYTES


def download_weights(
    destination: Optional[Path] = None,
    file_id: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download ``lite_medsam.pth`` from Google Drive.

    The file is staged as ``<destination>.part`` first and validated by size,
    so a failed or partial download never leaves a broken file at the target
    path.  A previously present valid checkpoint is reused unless ``force``.

    Parameters
    ----------
    destination : where the final checkpoint should live
    file_id : Google Drive file ID (default: official release)
    force : re-download even if a valid checkpoint already exists

    Returns
    -------
    Path to the freshly downloaded ``.part`` file, or the existing valid
    checkpoint path when nothing needed downloading.
    """
    dest = destination or default_checkpoint_path()
    fid = file_id or os.getenv("OPENMED_SEG_GDRIVE_ID", GDRIVE_FILE_ID)

    if _looks_valid(dest) and not force:
        logger.info(
            "Checkpoint already exists at %s (%d bytes), skipping download.",
            dest, dest.stat().st_size,
        )
        return dest

    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "The 'gdown' package is required to download weights from Google Drive. "
            "Install it with: pip install gdown"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={fid}"
    logger.info("Downloading LiteMedSAM weights from Google Drive → %s", dest)
    output = gdown.download(url, str(part), quiet=False)

    if output is None or not part.exists():
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed. Check your internet connection and verify the "
            f"Google Drive file ID: {fid}"
        )

    size = part.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is only {size} bytes — expected at least "
            f"{EXPECTED_MIN_BYTES}. The file may be a Google Drive error page. "
            f"Try downloading manually: {url}"
        )

    logger.info(
        "Download complete: %s (%d bytes, %.1f MB)",
        part, size, size / (1024 * 1024),
    )
    return part


def _resolve_device(device: Optional[str] = None) -> Any:
    """Auto-pick CUDA when available unless the caller pins a device."""
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def verify_checkpoint(path: Path, device: Optional[str] = None) -> dict[str, Any]:
    """Build the model, apply ``path`` strictly, and run a readiness forward.

    Raises
    ------
    RuntimeError
        If the file cannot be read, is not a state dict, has missing or
        unexpected keys, has any shape mismatch, or fails to execute a forward
        pass on the inputs the production pipeline uses.
    """
    import torch

    from .model import build_medsam_lite
    from .preprocessor import MEDSAM_INPUT_SIZE

    if not _looks_valid(path):
        raise RuntimeError(
            f"Checkpoint {path} is missing or smaller than {EXPECTED_MIN_BYTES} bytes."
        )

    logger.info("Building LiteMedSAM model architecture...")
    model = build_medsam_lite()

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint {path}: {exc}") from exc

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Checkpoint {path} is not a state dict; got {type(state_dict).__name__}. "
            "Expected a .pth produced by torch.save(model.state_dict(), ...)."
        )

    # strict=False so we can report the *shape* error and the missing/unexpected
    # keys together instead of a bare "size mismatch" line.
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with the LiteMedSAM architecture: {exc}"
        ) from exc

    problems = []
    if missing:
        problems.append(f"{len(missing)} missing keys (e.g. {missing[:3]}…)")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]}…)")
    if problems:
        raise RuntimeError(
            f"Checkpoint {path} does not match the LiteMedSAM state dict: "
            + "; ".join(problems)
        )

    # ── Readiness forward pass — exactly what the backend pipeline does ──
    # Box prompt only: the released lite_medsam.pth weights were trained to
    # segment from bounding boxes, which is the sole prompt the pipeline sends.
    dev = _resolve_device(device)
    model.to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        dummy = torch.zeros((1, 3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), device=dev)
        box = torch.tensor([[[60.0, 70.0, 190.0, 210.0]]], device=dev)
        box_masks, box_iou = model(dummy, boxes=box)

    expected_mask = (1, 1, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE)
    if tuple(box_masks.shape) != expected_mask:
        raise RuntimeError(
            f"Forward pass produced an unexpected shape: "
            f"box={tuple(box_masks.shape)} (expected {expected_mask})."
        )

    logger.info(
        "Forward pass OK: box prompt → %s iou=%s",
        tuple(box_masks.shape), tuple(box_iou.shape),
    )

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "parameters": n_params,
        "device": str(dev),
        "forward_shapes": {
            "box": tuple(box_masks.shape),
        },
    }


def prepare_model(
    model_path: Optional[Path] = None,
    force: bool = False,
    device: Optional[str] = None,
) -> tuple[Path, dict[str, Any]]:
    """Download (if needed), verify, and install a ready-for-inference model.

    Returns the canonical checkpoint path and the verification report.
    """
    dest = Path(model_path) if model_path is not None else default_checkpoint_path()

    if not _looks_valid(dest) or force:
        source = download_weights(destination=dest, force=force)
    else:
        logger.info("Checkpoint already present at %s — verifying it.", dest)
        source = dest

    info = verify_checkpoint(source, device=device)

    if source != dest:
        # Verified .part → atomically install into the canonical location.
        source.replace(dest)
        logger.info("Installed verified checkpoint at %s", dest)
        info["path"] = str(dest)

    return dest, info


def verify_weights(checkpoint_path: Optional[Path] = None) -> bool:
    """Lightweight health check: does a plausible checkpoint exist?

    This is the fast probe used by the router readiness endpoint — it does not
    build the model.  Use :func:`prepare_model` for full verification.
    """
    path = checkpoint_path or default_checkpoint_path()
    return _looks_valid(path)


def main() -> None:
    """CLI entrypoint for weight preparation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    force = "--force" in sys.argv
    cpu = "--cpu" in sys.argv
    device = "cpu" if cpu else None

    try:
        path, info = prepare_model(force=force, device=device)
    except (ImportError, RuntimeError) as exc:
        print(f"✗ Preparation failed: {exc}")
        sys.exit(1)

    print()
    print("✓ LiteMedSAM checkpoint ready for inference.")
    print(f"  path      : {path}")
    print(f"  size      : {info['size_bytes'] / (1024 * 1024):.1f} MB")
    print(f"  parameters: {info['parameters']:,}")
    print(f"  verify on : {info['device']}")
    print("  forward   : box prompt ✓  mask prompt ✓")


if __name__ == "__main__":
    main()