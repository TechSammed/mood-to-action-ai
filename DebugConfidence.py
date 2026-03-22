import joblib
import numpy as np
import pandas as pd
from src.features import build_features
from src.models import load_models
from src.uncertainty import compute_uncertainty

text_vectorizer = joblib.load("models/text_vectorizer.pkl")
meta_cols       = joblib.load("models/meta_cols.pkl")
scaler          = joblib.load("models/scaler.pkl")
state_model, _  = load_models(tag="full")

df = pd.DataFrame([{
    "id": 0,
    "journal_text": "the ocean sounds completely settled my mind. i feel peaceful relaxed and ready. my breathing slowed and shoulders feel light",
    "ambience_type": "ocean", "duration_min": 30,
    "sleep_hours": 8.0, "energy_level": 5, "stress_level": 1,
    "time_of_day": "morning", "previous_day_mood": "calm",
    "face_emotion_hint": "calm_face", "reflection_quality": "clear"
}])

X, _, _, _, _ = build_features(df, fit=False, text_vectorizer=text_vectorizer, meta_cols=meta_cols, scaler=scaler)
_, proba = state_model.predict_and_proba(X)

print("\n── Probabilities ──────────────────────")
for state, p in zip(state_model.classes_, proba[0]):
    print(f"  {state:15s}: {p:.3f}")

print(f"\nMax prob      : {proba[0].max():.3f}")
print(f"Top 2 gap     : {sorted(proba[0])[-1] - sorted(proba[0])[-2]:.3f}")

text_len = len(df["journal_text"][0])
print(f"\nText length   : {text_len} chars")

conf, flag = compute_uncertainty(proba, text_lengths=np.array([text_len]))
print(f"Confidence    : {conf[0]:.3f}")
print(f"Uncertain flag: {flag[0]}")
print("───────────────────────────────────────")