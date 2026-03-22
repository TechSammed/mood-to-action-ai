"""
src/uncertainty.py
------------------
Confidence score + uncertain_flag.

Fixed for small dataset (1080 rows, 6 classes) where max_prob
naturally sits around 0.30-0.45 due to class similarity.

Formula (revised):
    confidence = (0.60 * max_prob + 0.40 * margin) * quality_mult
    then rescaled to [0, 1] using dataset-realistic bounds

uncertain_flag = 1 if confidence < 0.35
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Tuple, Optional

UNCERTAINTY_THRESHOLD = 0.35
SHORT_TEXT_THRESHOLD  = 20
MEDIUM_TEXT_THRESHOLD = 80

# Realistic bounds for this dataset (6 classes, small, noisy)
# Max prob realistically ranges from 0.25 (random) to 0.75 (very clear)
MIN_EXPECTED_PROB = 0.167   # 1/6 = pure random
MAX_EXPECTED_PROB = 0.80    # near certain


def _rescale_prob(max_prob: np.ndarray) -> np.ndarray:
    """Rescale max_prob from realistic range to [0, 1]."""
    rescaled = (max_prob - MIN_EXPECTED_PROB) / (MAX_EXPECTED_PROB - MIN_EXPECTED_PROB)
    return np.clip(rescaled, 0.0, 1.0)


def _prediction_margin(proba: np.ndarray) -> np.ndarray:
    """Gap between top-2 probabilities — rescaled to [0, 1]."""
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    margin   = sorted_p[:, 0] - sorted_p[:, 1]
    # Max possible margin in 6-class = ~0.83, rescale
    return np.clip(margin / 0.60, 0.0, 1.0)


def _text_quality_multiplier(
    text_lengths: np.ndarray,
    reflection_quality: Optional[np.ndarray] = None,
) -> np.ndarray:
    n       = len(text_lengths)
    quality = np.ones(n, dtype=float)

    # Text length penalty
    short_mask  = text_lengths < SHORT_TEXT_THRESHOLD
    medium_mask = (text_lengths >= SHORT_TEXT_THRESHOLD) & (text_lengths < MEDIUM_TEXT_THRESHOLD)
    quality[short_mask]  = 0.60
    quality[medium_mask] = 0.85

    # Reflection quality label
    if reflection_quality is not None:
        rq  = np.array(reflection_quality, dtype=float)
        rq_mult = 0.80 + 0.10 * np.clip(rq, 0, 2)  # conflicted=0.80, vague=0.90, clear=1.0
        quality = quality * rq_mult

    return np.clip(quality, 0.50, 1.0)


def compute_uncertainty(
    state_proba: np.ndarray,
    text_lengths: np.ndarray,
    reflection_quality: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    max_prob      = np.max(state_proba, axis=1)
    rescaled_prob = _rescale_prob(max_prob)
    margin        = _prediction_margin(state_proba)
    quality_mult  = _text_quality_multiplier(text_lengths, reflection_quality)

    # Weighted combination
    raw = (0.60 * rescaled_prob + 0.40 * margin) * quality_mult

    confidence     = np.clip(raw, 0.05, 0.98).astype(np.float32)
    uncertain_flag = (confidence < UNCERTAINTY_THRESHOLD).astype(int)

    return confidence, uncertain_flag