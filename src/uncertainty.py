"""
src/uncertainty.py
------------------
Confidence score + uncertain_flag.

confidence = 0.50 × max_prob + 0.25 × entropy_penalty + 0.25 × margin_certainty
             × text_quality_multiplier

uncertain_flag = 1 if confidence < 0.45
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Tuple, Optional

UNCERTAINTY_THRESHOLD = 0.45
SHORT_TEXT_THRESHOLD  = 20
MEDIUM_TEXT_THRESHOLD = 80


def _prediction_entropy(proba: np.ndarray) -> np.ndarray:
    n_classes = proba.shape[1]
    raw = np.array([scipy_entropy(p + 1e-10) for p in proba])
    return raw / (np.log(n_classes) + 1e-10)


def _prediction_margin(proba: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    margin = sorted_proba[:, 0] - sorted_proba[:, 1]
    return 1.0 - margin


def _text_quality_penalty(
    text_lengths: np.ndarray,
    reflection_quality: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(text_lengths)
    penalty = np.ones(n, dtype=float)

    short_mask  = text_lengths < SHORT_TEXT_THRESHOLD
    medium_mask = (text_lengths >= SHORT_TEXT_THRESHOLD) & (text_lengths < MEDIUM_TEXT_THRESHOLD)
    penalty[short_mask]  = 0.55
    penalty[medium_mask] = 0.80

    if reflection_quality is not None:
        rq = np.array(reflection_quality, dtype=float)
        quality_multiplier = 0.7 + 0.1 * np.clip(rq, 0, 3)
        penalty = penalty * quality_multiplier

    return np.clip(penalty, 0.5, 1.0)


def compute_uncertainty(
    state_proba: np.ndarray,
    text_lengths: np.ndarray,
    reflection_quality: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    max_prob         = np.max(state_proba, axis=1)
    entropy_penalty  = 1.0 - _prediction_entropy(state_proba)
    margin_certainty = 1.0 - _prediction_margin(state_proba)
    quality_mult     = _text_quality_penalty(text_lengths, reflection_quality)

    raw_confidence = (
        0.50 * max_prob +
        0.25 * entropy_penalty +
        0.25 * margin_certainty
    ) * quality_mult

    confidence     = np.clip(raw_confidence, 0.05, 0.98).astype(np.float32)
    uncertain_flag = (confidence < UNCERTAINTY_THRESHOLD).astype(int)

    return confidence, uncertain_flag