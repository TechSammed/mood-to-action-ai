"""
api.py — FastAPI inference server
Run:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
import pandas as pd
import traceback
import joblib

from src.features import build_features
from src.models import load_models
from src.decision import decide, generate_supportive_message
from src.uncertainty import compute_uncertainty

app = FastAPI(
    title="ArvyaX Mental State API",
    description="Predicts emotional state, intensity, and recommends a guided action.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state_model = intensity_model = text_vectorizer = meta_cols = scaler = None

@app.on_event("startup")
async def load_artifacts():
    global state_model, intensity_model, text_vectorizer, meta_cols, scaler
    try:
        state_model, intensity_model = load_models(tag="full")
        text_vectorizer = joblib.load("models/text_vectorizer.pkl")
        meta_cols       = joblib.load("models/meta_cols.pkl")
        scaler          = joblib.load("models/scaler.pkl")
        print("✓ Models loaded.")
    except Exception as e:
        print(f"⚠ Model load failed: {e}. Run python main.py first.")


class PredictRequest(BaseModel):
    journal_text:       str   = Field(..., example="Feeling restless after the rain session.")
    ambience_type:      Optional[str]   = Field("rain",    example="rain")
    duration_min:       Optional[float] = Field(20,        example=20)
    sleep_hours:        Optional[float] = Field(7.0,       example=6.5)
    energy_level:       Optional[float] = Field(3.0,       example=3, ge=1, le=5)
    stress_level:       Optional[float] = Field(3.0,       example=4, ge=1, le=5)
    time_of_day:        Optional[Literal["early_morning","morning","afternoon","evening","night"]] = "morning"
    previous_day_mood:  Optional[str]   = Field("neutral", example="calm")
    face_emotion_hint:  Optional[str]   = Field("neutral_face", example="tense_face")
    reflection_quality: Optional[Literal["clear","vague","conflicted"]] = "vague"


class PredictResponse(BaseModel):
    predicted_state:     str
    predicted_intensity: int
    confidence:          float
    uncertain_flag:      int
    what_to_do:          str
    when_to_do:          str
    message:             str


def _build_df(req: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": 0, "journal_text": req.journal_text,
        "ambience_type": req.ambience_type, "duration_min": req.duration_min,
        "sleep_hours": req.sleep_hours, "energy_level": req.energy_level,
        "stress_level": req.stress_level, "time_of_day": req.time_of_day,
        "previous_day_mood": req.previous_day_mood,
        "face_emotion_hint": req.face_emotion_hint,
        "reflection_quality": req.reflection_quality,
    }])


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "models_loaded": state_model is not None, "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    if state_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run python main.py first.")
    try:
        df = _build_df(req)
        X_full, _, _, _, _ = build_features(
            df, fit=False, text_vectorizer=text_vectorizer,
            meta_cols=meta_cols, scaler=scaler,
        )
        pred_state, proba    = state_model.predict_and_proba(X_full)
        pred_intensity       = intensity_model.predict(X_full)
        confidence, unc_flag = compute_uncertainty(proba, text_lengths=df["journal_text"].apply(len).values)
        what_list, when_list = decide(
            pred_state, pred_intensity,
            df["stress_level"].values, df["energy_level"].values,
            df["time_of_day"].values,  df["sleep_hours"].values,
        )
        message = generate_supportive_message(
            pred_state[0], pred_intensity[0], what_list[0], when_list[0], float(confidence[0])
        )
        return {
            "predicted_state": pred_state[0], "predicted_intensity": int(pred_intensity[0]),
            "confidence": round(float(confidence[0]), 3), "uncertain_flag": int(unc_flag[0]),
            "what_to_do": what_list[0], "when_to_do": when_list[0], "message": message,
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(requests: list[PredictRequest]):
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100.")
    return [await predict(r) for r in requests]


@app.get("/states", tags=["Meta"])
async def list_states():
    if state_model and hasattr(state_model, "classes_"):
        return {"states": list(state_model.classes_)}
    return {"states": ["calm", "focused", "neutral", "restless", "overwhelmed", "mixed"]}