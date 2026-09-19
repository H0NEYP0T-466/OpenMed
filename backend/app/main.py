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
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.organs.brain.classification.router import router as brain_router


# ── Logging Setup ─────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    """ANSI-coloured formatter — no external deps."""

    _COLORS = {
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

app = FastAPI(
    title="OpenMed API",
    description="AI Hospital — Brain Tumor Classification & more",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(brain_router, prefix="/api/brain", tags=["Brain"])


@app.on_event("startup")
async def _startup() -> None:
    logger.info("━" * 60)
    logger.info("  OpenMed FastAPI Server  ·  starting up")
    logger.info("━" * 60)
    logger.info("  Brain classification : /api/brain/classify")
    logger.info("  Brain model info     : /api/brain/model-info")
    logger.info("  Brain health         : /api/brain/health")
    logger.info("━" * 60)


@app.get("/")
def read_root():
    return {
        "service": "OpenMed API",
        "status": "running",
        "organs": ["brain"],
    }


# ── Main entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8016,
        reload=True,
        log_level="info",
    )
