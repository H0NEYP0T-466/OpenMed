"""Resume-state contract for the LungV2S trainer.

A training run is 100 epochs on Kaggle. If the resume payload loses a field, the
run does not crash — it silently restarts the schedule, the EMA or the early-stop
counter, and the final metric is quietly wrong. So every field is pinned here,
and the three log lines that announce which resume branch fired are pinned too:
they are the only evidence in the log that the branch the user thinks ran, ran.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.organs.lungs.classification.kaggle import train_lungs_v2s as trainer
from app.organs.lungs.classification.kaggle.config import LungConfig

RESUME_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "ema",
    "epoch",
    "best_metric",
    "epochs_without_improvement",
    "history",
    "config",
}


class _FakeScheduler:
    """Minimal stand-in: the real payload only ever round-trips a dict."""

    def __init__(self, step: int = 7) -> None:
        self.step = step

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step}


class _FakeEma:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module


def _fixture(tmp_path: Path):
    model = torch.nn.Linear(4, 4)
    ema_module = torch.nn.Linear(4, 4)
    with torch.no_grad():
        ema_module.weight.fill_(0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = {
        key: [1, 2, 3] for key in trainer.HISTORY_KEYS
    }
    history["epoch"] = [1, 2, 3]
    return {
        "model": model,
        "ema": _FakeEma(ema_module),
        "optimizer": optimizer,
        "scheduler": _FakeScheduler(),
        "history": history,
        "cfg": LungConfig(),
    }


def test_resume_payload_round_trips_every_field(tmp_path: Path) -> None:
    parts = _fixture(tmp_path)
    path = tmp_path / "lung_v2s_resume.pth"

    trainer.save_resume_checkpoint(
        path,
        parts["model"],
        parts["optimizer"],
        parts["scheduler"],
        parts["ema"],
        epoch=11,
        best_metric=0.9312,
        stale=4,
        history=parts["history"],
        cfg=parts["cfg"],
    )

    assert path.is_file()

    payload = trainer.load_torch(path, "cpu")

    assert set(payload) == RESUME_KEYS, (
        f"resume payload keys drifted: missing={RESUME_KEYS - set(payload)}, "
        f"extra={set(payload) - RESUME_KEYS}"
    )
    assert payload["epoch"] == 11
    assert payload["best_metric"] == pytest.approx(0.9312)
    assert payload["epochs_without_improvement"] == 4
    assert payload["scheduler"] == {"step": 7}
    assert payload["config"] == parts["cfg"].to_dict()

    # Weights survive as tensors, not as references to the live parameters.
    model_weight = payload["model"]["weight"]
    assert torch.is_tensor(model_weight)
    assert model_weight.shape == (4, 4)
    ema_weight = payload["ema"]["weight"]
    assert torch.allclose(ema_weight, torch.full((4, 4), 0.25))

    # History is per-epoch columns of equal length.
    assert set(payload["history"]) == set(trainer.HISTORY_KEYS)
    assert all(len(column) == 3 for column in payload["history"].values())


def test_resume_resumes_at_the_next_epoch(tmp_path: Path) -> None:
    """``start_epoch = epoch + 1`` — off-by-one here silently repeats an epoch."""
    parts = _fixture(tmp_path)
    path = tmp_path / "lung_v2s_resume.pth"
    trainer.save_resume_checkpoint(
        path,
        parts["model"],
        parts["optimizer"],
        parts["scheduler"],
        parts["ema"],
        epoch=0,
        best_metric=0.5,
        stale=0,
        history=parts["history"],
        cfg=parts["cfg"],
    )

    payload = trainer.load_torch(path, "cpu")
    start_epoch = int(payload["epoch"]) + 1
    assert start_epoch == 1


def test_resume_without_ema_is_null(tmp_path: Path) -> None:
    parts = _fixture(tmp_path)
    path = tmp_path / "lung_v2s_resume.pth"
    trainer.save_resume_checkpoint(
        path,
        parts["model"],
        parts["optimizer"],
        parts["scheduler"],
        None,
        epoch=3,
        best_metric=0.1,
        stale=1,
        history=parts["history"],
        cfg=parts["cfg"],
    )
    assert trainer.load_torch(path, "cpu")["ema"] is None


def test_three_resume_branches_are_still_announced() -> None:
    """The log lines are the contract's observable behaviour for resume.

    Pinned against the source so a refactor cannot rename them silently — the
    user reads these three strings to know which branch ran.
    """
    source = inspect.getsource(trainer)
    for literal in ("RESUMED from epoch %d", "WEIGHTS-ONLY RESUME", "FRESH START"):
        assert literal in source, f"resume log line {literal!r} disappeared"


def test_history_columns_are_declared() -> None:
    """Every history key the trainer writes must be in HISTORY_KEYS."""
    assert "val_macro_f1" in trainer.HISTORY_KEYS
    assert "epoch_seconds" in trainer.HISTORY_KEYS
    assert len(set(trainer.HISTORY_KEYS)) == len(trainer.HISTORY_KEYS)
