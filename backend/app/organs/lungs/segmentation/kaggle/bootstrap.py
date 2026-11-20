"""Stage 1: environment report, dependency check, hardware assertion, encoder self-check."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
from typing import Any

logger = logging.getLogger("bootstrap")

REQUIRED = ("torch", "numpy", "cv2", "scipy", "matplotlib")


def check_dependencies(need_transformers: bool) -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    names = REQUIRED + (("transformers",) if need_transformers else ())
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing Python packages: " + ", ".join(missing)
            + ". On Kaggle: !pip install transformers opencv-python-headless scipy matplotlib"
        )
    return versions


def environment_report() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    gpus = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": props.name,
                         "memory_gb": round(props.total_memory / 2**30, 1),
                         "capability": f"{props.major}.{props.minor}"})
    disk = shutil.disk_usage(os.getcwd())
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu_count": len(gpus), "gpus": gpus, "gpu_name": gpus[0]["name"] if gpus else "cpu",
        "cpu_count": os.cpu_count(), "disk_free_gb": round(disk.free / 2**30, 1),
        "on_kaggle": os.path.isdir("/kaggle/input"),
    }


def log_environment(env: dict[str, Any]) -> None:
    logger.info("python %s | torch %s | cuda %s | cudnn %s", env["python"], env["torch"],
                env["cuda"], env["cudnn"])
    logger.info("cpus %s | free disk %.1f GB | kaggle %s", env["cpu_count"], env["disk_free_gb"],
                env["on_kaggle"])
    for gpu in env["gpus"]:
        logger.info("GPU %d: %s %.1f GB (sm_%s)", gpu["index"], gpu["name"], gpu["memory_gb"],
                    gpu["capability"].replace(".", ""))
    if not env["gpus"]:
        logger.warning("no CUDA device visible")


def resolve_gpus(requested: int, env: dict[str, Any], *, allow_cpu: bool) -> int:
    """Clamp the request to what exists; refuse a silent CPU run of the real job."""
    available = env["gpu_count"]
    if available == 0:
        if allow_cpu:
            logger.warning("running on CPU: audit or wiring check only, never the real job")
            return 0
        raise RuntimeError(
            "No GPU visible. In the Kaggle notebook set Accelerator to 'GPU T4 x2' (Settings -> "
            "Accelerator) and rerun, rather than training for hours on CPU. --audit-only needs no GPU."
        )
    if requested > available:
        logger.warning("requested %d GPUs but %d visible; using %d", requested, available, available)
        return available
    if requested < 1:
        return 1
    return requested


def encoder_selfcheck() -> dict[str, Any]:
    """Prove the installed ``transformers`` layout round-trips through the canonical naming.

    Builds a random MiT-B0, exports its weights under the canonical names, loads them
    into a second instance with a strict key check, and requires identical outputs.
    A layout this build cannot translate fails here, at stage 1, not after hours.
    """
    import torch

    from ..model import SegSpec, build_model

    spec = SegSpec(drop_path=0.0, decoder_dropout=0.0, head_dropout=0.0)
    torch.manual_seed(0)
    first = build_model(spec).eval()
    second = build_model(spec).eval()
    state = first.canonical_state_dict()
    second.load_canonical(state)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        a, _ = first(x)
        b, _ = second(x)
    diff = float((a - b).abs().max())
    if diff > 1e-5:
        raise RuntimeError(f"Encoder canonical round-trip changed the output (max |diff| = {diff:.2e}).")
    layout = first.encoder.layout
    logger.info("encoder self-check: layout %r, %d canonical tensors, round-trip max |diff| %.1e",
                layout, len(state), diff)
    return {"layout": layout, "tensors": len(state), "round_trip_max_abs_diff": diff}
