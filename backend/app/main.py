"""
OpenMed — AI Hospital  ·  FastAPI Backend Server

Entrypoint for the FastAPI application.  Run with:
    python -m app.main          (from backend/)
    uvicorn app.main:app --reload --port 8016
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, ClassVar

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.organs.brain.classification.router import checkpoint_path
from app.organs.brain.classification.router import router as brain_router

# ── Logging Setup ─────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    """ANSI-coloured formatter — no external deps."""

    _COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[2;37m",       # dim white
        "INFO": "\033[92m",          # green
        "WARNING": "\033[93m",       # yellow
        "ERROR": "\033[91m",         # red
        "CRITICAL": "\033[1;91m",    # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        c = self._COLORS.get(record.levelname, self._RESET)
        record.levelname = f"{c}{record.levelname:<8}{self._RESET}"
        return super().format(record)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    _ColorFormatter(
        "[%(asctime)s] %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
)
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)

# Quiet noisy third-party loggers
for _name in ("uvicorn.access", "PIL", "matplotlib"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger("openmed")


# ── FastAPI App ───────────────────────────────────────────────────────────

DEFAULT_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _cors_origins() -> list[str]:
    configured = os.getenv("OPENMED_CORS_ORIGINS", "").strip()
    if not configured:
        return list(DEFAULT_DEV_ORIGINS)
    if configured == "*":
        logger.warning("CORS opened to all origins; credentials will be disabled.")
        return ["*"]
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    logger.info("=" * 62)
    logger.info("  OpenMed FastAPI server")
    logger.info("  classify     : POST /api/brain/classify")
    logger.info("  model info   : GET  /api/brain/model-info")
    logger.info("  health       : GET  /api/brain/health")
    logger.info("  checkpoint   : %s", checkpoint_path())
    logger.info("  cors origins : %s", ", ".join(_cors_origins()))
    logger.info("=" * 62)
    yield


app = FastAPI(
    title="OpenMed API",
    description="AI Hospital — Brain Tumor Classification & more",
    version="0.1.0",
    lifespan=_lifespan,
)

_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(brain_router, prefix="/api/brain", tags=["Brain"])


@app.get("/")
def read_root():
    return {
        "service": "OpenMed API",
        "status": "running",
        "organs": ["brain"],
        "endpoints": [
            "/api/brain/classify",
            "/api/brain/model-info",
            "/api/brain/health",
        ],
    }


# ── Main entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("OPENMED_HOST", "0.0.0.0"),
        port=int(os.getenv("OPENMED_PORT", "8016")),
        reload=os.getenv("OPENMED_RELOAD", "1") == "1",
        log_level="info",
    )
