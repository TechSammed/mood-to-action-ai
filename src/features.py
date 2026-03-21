"""
src/features.py
---------------
Feature engineering — built from actual dataset column values.

Real dataset facts:
  - emotional_state : calm, focused, neutral, restless, overwhelmed, mixed
  - reflection_quality : clear, vague, conflicted
  - time_of_day : morning, early_morning, afternoon, evening, night
  - previous_day_mood : calm, focused, neutral, overwhelmed, mixed, restless
  - face_emotion_hint : calm_face, neutral_face, tense_face, tired_face, happy_face, none/NaN
  - energy_level / stress_level : scale 1–5
  - sleep_hours : ~3.5–8.5
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline as SKPipeline
import re
from typing import Tuple, Optional, List

# ── Optional heavy deps ────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    USE_SBERT = True
    print("  [features] Using sentence-transformers (all-MiniLM-L6-v2)")
except ImportError:
    USE_SBERT = False
    print("  [features] sentence-transformers not found — using TF-IDF+SVD fallback")

try:
    from textblob import TextBlob
    USE_TEXTBLOB = True
except ImportError:
    USE_TEXTBLOB = False

# ── Ordinal maps (from real dataset values) ────────────────────────────────
TIME_MAP = {
    "early_morning": 0, "morning": 1, "afternoon": 2, "evening": 3, "night": 4
}
MOOD_MAP = {
    "overwhelmed": 0, "restless": 1, "mixed": 2,
    "neutral": 3, "calm": 4, "focused": 5
}
QUALITY_MAP = {
    "conflicted": 0, "vague": 1, "clear": 2
}
EMOTION_MAP = {
    "tired_face": 0, "tense_face": 1, "neutral_face": 2,
    "calm_face": 3, "happy_face": 4
}
AMBIENCE_MAP = {
    "forest": 0, "ocean": 1, "rain": 2, "mountain": 3, "cafe": 4
}

META_COLS = [
    "sleep_hours", "stress_level", "energy_level", "duration_min",
    "time_of_day_enc", "previous_day_mood_enc",
    "face_emotion_hint_enc", "reflection_quality_enc", "ambience_enc"
]


def _clean_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "no reflection provided"
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_text_stats(texts: List[str]) -> np.ndarray:
    """Handcrafted signals: word count, char count, sentiment, punctuation."""
    rows = []
    for t in texts:
        if not isinstance(t, str) or t.strip() == "":
            rows.append([0, 0, 0, 0, 0, 0, 0])
            continue
        words  = t.split()
        wc     = len(words)
        chars  = len(t)
        avg_wl = np.mean([len(w) for w in words]) if words else 0
        excl   = t.count("!")
        quest  = t.count("?")
        upper_r = sum(1 for c in t if c.isupper()) / max(chars, 1)
        pol = 0.0
        if USE_TEXTBLOB:
            try:
                pol = TextBlob(t).sentiment.polarity
            except Exception:
                pol = 0.0
        rows.append([wc, chars, avg_wl, excl, quest, upper_r, pol])
    return np.array(rows, dtype=np.float32)


def _encode_meta(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    d = df.copy()

    # Ordinal encodes — using real dataset values
    d["time_of_day_enc"]        = d["time_of_day"].str.lower().map(TIME_MAP).fillna(2).astype(float)
    d["previous_day_mood_enc"]  = d["previous_day_mood"].str.lower().map(MOOD_MAP).fillna(2).astype(float)
    d["reflection_quality_enc"] = d["reflection_quality"].str.lower().map(QUALITY_MAP).fillna(1).astype(float)
    d["face_emotion_hint_enc"]  = d["face_emotion_hint"].str.lower().map(EMOTION_MAP).fillna(2).astype(float)
    d["ambience_enc"]           = d["ambience_type"].str.lower().map(AMBIENCE_MAP).fillna(2).astype(float)

    # Numeric — scale is 1–5 for stress/energy
    d["sleep_hours"]  = d["sleep_hours"].fillna(d["sleep_hours"].median()).clip(0, 12)
    d["stress_level"] = d["stress_level"].fillna(d["stress_level"].median()).clip(1, 5)
    d["energy_level"] = d["energy_level"].fillna(d["energy_level"].median()).clip(1, 5)
    d["duration_min"] = d["duration_min"].fillna(d["duration_min"].median()).clip(0, 120)

    # Derived features
    d["energy_stress_ratio"] = d["energy_level"] / (d["stress_level"] + 0.5)
    d["sleep_deficit"]       = (d["sleep_hours"] < 6).astype(float)
    d["high_stress_flag"]    = (d["stress_level"] >= 4).astype(float)
    d["low_energy_flag"]     = (d["energy_level"] <= 2).astype(float)

    all_cols = META_COLS + ["energy_stress_ratio", "sleep_deficit", "high_stress_flag", "low_energy_flag"]
    return d[all_cols].values.astype(np.float32), all_cols


class TextVectorizer:
    """SBERT or TF-IDF+SVD fallback."""

    def __init__(self):
        self.use_sbert = USE_SBERT
        if not self.use_sbert:
            self._pipe = SKPipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000, ngram_range=(1, 2),
                    sublinear_tf=True, min_df=2
                )),
                ("svd", TruncatedSVD(n_components=200, random_state=42))
            ])

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.use_sbert:
            return _ST_MODEL.encode(texts, show_progress_bar=True, batch_size=64)
        return self._pipe.fit_transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.use_sbert:
            return _ST_MODEL.encode(texts, show_progress_bar=False, batch_size=64)
        return self._pipe.transform(texts)


def build_features(
    df: pd.DataFrame,
    fit: bool = True,
    text_vectorizer: Optional[TextVectorizer] = None,
    meta_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], TextVectorizer, StandardScaler]:
    """
    Returns: X_full, X_text_only, meta_cols, text_vectorizer, scaler
    """
    texts = df["journal_text"].fillna("").apply(_clean_text).tolist()

    # Text embedding
    if fit:
        text_vectorizer = TextVectorizer()
        text_feats = text_vectorizer.fit_transform(texts)
    else:
        text_feats = text_vectorizer.transform(texts)

    # Handcrafted text stats
    text_stats = _extract_text_stats(texts)

    # Metadata
    meta_feats, meta_col_names = _encode_meta(df)

    # Scale metadata
    if fit:
        scaler = StandardScaler()
        meta_feats = scaler.fit_transform(meta_feats)
    else:
        meta_feats = scaler.transform(meta_feats)

    X_full      = np.hstack([text_feats, text_stats, meta_feats]).astype(np.float32)
    X_text_only = np.hstack([text_feats, text_stats]).astype(np.float32)

    return X_full, X_text_only, meta_col_names, text_vectorizer, scaler