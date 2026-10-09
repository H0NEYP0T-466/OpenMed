"""Stage 1 — environment report, dependency check and hardware assertion."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
from typing import Any

logger = logging.getLogger("bootstrap")

REQUIRED = ("torch", "numpy", "cv2", "scipy", "matplotlib", "sklearn")


def check_dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing Python packages: " + ", ".join(missing)
            + ". On Kaggle: !pip install opencv-python-headless scikit-learn scipy matplotlib"
        )
    return versions


def environment_report() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    gpus = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index, "name": props.name,
                "memory_gb": round(props.total_memory / 2**30, 1),
                "capability": f"{props.major}.{props.minor}",
            })
    disk = shutil.disk_usage(os.getcwd())
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu_count": len(gpus), "gpus": gpus,
        "gpu_name": gpus[0]["name"] if gpus else "cpu",
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
    """Clamp the GPU request to what exists; refuse a silent CPU run of the real job."""
    available = env["gpu_count"]
    if available == 0:
        if allow_cpu:
            logger.warning("running on CPU (--allow-cpu / --smoke): wiring check only")
            return 0
        raise RuntimeError(
            "No GPU visible. In the Kaggle notebook set Accelerator to 'GPU T4 x2' (Settings -> "
            "Accelerator) and rerun, rather than training for hours on CPU. Use --smoke to test wiring."
        )
    if requested > available:
        logger.warning("requested %d GPUs but %d visible — using %d", requested, available, available)
        return available
    return max(1, requested)
