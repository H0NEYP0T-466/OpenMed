"""Locating the MedSAM2 checkpoint.

Weights are never committed and never downloaded implicitly at request time: a
tab click should fail with a sentence, not a 400 MB surprise. `ensure()` is the
only entry point and it returns a path or raises with instructions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = PACKAGE_DIR / "weights"
DEFAULT_FILENAME = "MedSAM2_latest.pt"
REPO_ID = "wanglab/MedSAM2"
DOWNLOAD_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{DEFAULT_FILENAME}"
# Anything smaller than this is an HTML error page, not a checkpoint.
MIN_PLAUSIBLE_BYTES = 10 * 1024 * 1024


def checkpoint_path() -> Path:
    override = os.getenv("OPENMED_MEDSAM2_CHECKPOINT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_WEIGHTS_DIR / DEFAULT_FILENAME


def status() -> tuple[bool, str]:
    """(present_and_plausible, human reason)."""
    path = checkpoint_path()
    if not path.is_file():
        return False, (
            f"MedSAM2 weights not installed. Expected {path}. "
            f"Download {DOWNLOAD_URL} and place it there, or set "
            f"OPENMED_MEDSAM2_CHECKPOINT to its path."
        )
    size = path.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        return False, (
            f"{path.name} is {size} bytes — that is an error page, not a checkpoint. "
            f"Re-download {DOWNLOAD_URL}."
        )
    return True, ""


def ensure() -> Path:
    ok, reason = status()
    if not ok:
        raise FileNotFoundError(reason)
    return checkpoint_path()


def download(destination: Path | None = None, *, token: str | None = None) -> Path:
    """Fetch the checkpoint once, into the package's weights folder.

    Exposed for operators (`python -m app.experimental.medsam2.weights`) rather
    than called by the API. HuggingFace rate-limits aggressively; a failure here
    is reported, never retried in a loop.
    """
    from huggingface_hub import hf_hub_download

    target = destination or checkpoint_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s → %s", DEFAULT_FILENAME, target)
    fetched = hf_hub_download(
        repo_id=REPO_ID,
        filename=DEFAULT_FILENAME,
        token=token,
        local_dir=str(target.parent),
    )
    resolved = Path(fetched)
    if resolved != target:
        resolved.replace(target)
    ok, reason = status()
    if not ok:
        raise RuntimeError(reason)
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(download())
