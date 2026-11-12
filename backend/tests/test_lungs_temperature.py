"""Unit tests for temperature-scaling calibration (CPU-only, no model weights)."""

from __future__ import annotations

import math

import numpy as np

from app.organs.lungs.classification.kaggle import fit_temperature as ft


def _build_logits(
    confidence: np.ndarray, prediction: np.ndarray, num_classes: int = 4
) -> np.ndarray:
    """Logits whose softmax top probability is ``confidence`` at ``prediction``.

    All other classes share an equal, lower logit, so the top-class softmax is
    ``c = e^a / (e^a + (K-1))``. Solving for ``a`` gives the expression below.
    """
    logits = np.zeros((len(confidence), num_classes), dtype=np.float64)
    for i in range(len(confidence)):
        c = min(max(float(confidence[i]), 1e-4), 1.0 - 1e-4)
        logits[i, int(prediction[i])] = math.log(c * (num_classes - 1) / (1.0 - c))
    return logits.astype(np.float32)


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    probs = ft._softmax(logits, 1.0)
    assert probs.shape == logits.shape
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-6)


def test_ece_near_zero_when_calibrated() -> None:
    # Confidence 0.70, and correct 70% of the time -> calibrated.
    n = 5000
    confidence = np.full(n, 0.70)
    prediction = np.zeros(n, dtype=int)
    prediction[int(0.70 * n) :] = 1  # last 30% wrong
    labels = np.zeros(n, dtype=np.int64)  # true class is always 0
    probs = ft._softmax(_build_logits(confidence, prediction), 1.0)
    ece = ft.expected_calibration_error(probs, labels)
    assert ece < 0.05


def test_fit_temperature_shrinks_overconfidence() -> None:
    # Says 0.95, right only 60% of the time -> needs T > 1 (soften).
    n = 4000
    confidence = np.full(n, 0.95)
    prediction = np.zeros(n, dtype=int)
    prediction[int(0.60 * n) :] = 1
    labels = np.zeros(n, dtype=np.int64)
    logits = _build_logits(confidence, prediction)

    temperature = ft.fit_temperature(logits, labels)
    assert temperature > 1.0

    ece_before = ft.expected_calibration_error(ft._softmax(logits, 1.0), labels)
    ece_after = ft.expected_calibration_error(ft._softmax(logits, temperature), labels)
    assert ece_after < ece_before


def test_fit_temperature_sharpens_underconfidence() -> None:
    # Says 0.55, right 90% of the time -> needs T < 1 (sharpen).
    n = 4000
    confidence = np.full(n, 0.55)
    prediction = np.zeros(n, dtype=int)
    prediction[int(0.90 * n) :] = 1
    labels = np.zeros(n, dtype=np.int64)
    logits = _build_logits(confidence, prediction)

    temperature = ft.fit_temperature(logits, labels)
    assert 0.0 < temperature < 1.0


def test_fit_temperature_returns_positive_finite() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(200, 4)).astype(np.float32)
    labels = rng.integers(0, 4, size=200).astype(np.int64)
    temperature = ft.fit_temperature(logits, labels)
    assert np.isfinite(temperature)
    assert temperature > 0.0
