"""
ArvyaX Mental State ML Pipeline
================================
Entry point. Runs full pipeline: load → features → train → predict → output.

Usage:
    python main.py --train data/train.csv --test data/test.csv --out predictions.csv
    python main.py --train data/train.csv --test data/test.csv --ablation
"""

import argparse
import pandas as pd
import numpy as np
import warnings
import joblib
import os
warnings.filterwarnings("ignore")

from src.features import build_features, TextVectorizer
from src.models import train_emotional_state, train_intensity, load_models
from src.decision import decide
from src.uncertainty import compute_uncertainty
from src.utils import load_data, save_predictions


def run_pipeline(train_path: str, test_path: str, out_path: str, ablation: bool = False):
    print("\n=== ArvyaX Mental State Pipeline ===\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("[1/6] Loading data...")
    train_df = load_data(train_path)
    test_df  = load_data(test_path)
    print(f"      Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # ── 2. Feature engineering ─────────────────────────────────────────────
    print("[2/6] Building features...")
    X_train_full, X_train_text, meta_cols, text_vectorizer, scaler = build_features(
        train_df, fit=True
    )
    X_test_full, X_test_text, _, _, _ = build_features(
        test_df, fit=False,
        text_vectorizer=text_vectorizer,
        meta_cols=meta_cols,
        scaler=scaler
    )
    y_state     = train_df["emotional_state"]
    y_intensity = train_df["intensity"]
    print(f"      Feature dim (full): {X_train_full.shape[1]} | (text-only): {X_train_text.shape[1]}")

    # ── 3. Train models ────────────────────────────────────────────────────
    print("[3/6] Training models...")
    state_model     = train_emotional_state(X_train_full, y_state)
    intensity_model = train_intensity(X_train_full, y_intensity)

    # Save vectorizer + scaler for API / Streamlit use
    os.makedirs("models", exist_ok=True)
    joblib.dump(text_vectorizer, "models/text_vectorizer.pkl")
    joblib.dump(meta_cols,       "models/meta_cols.pkl")
    joblib.dump(scaler,          "models/scaler.pkl")
    print("      Saved vectorizer + scaler → models/")

    # Optional ablation
    if ablation:
        print("      [Ablation] Training text-only variants...")
        from src.ablation import run_ablation
        text_dim = X_train_text.shape[1]
        report = run_ablation(X_train_full, X_train_text, y_state, y_intensity, meta_cols, text_dim)
        print(report[:500])

    # ── 4. Predict ─────────────────────────────────────────────────────────
    print("[4/6] Generating predictions...")
    predicted_state, state_proba = state_model.predict_and_proba(X_test_full)
    predicted_intensity          = intensity_model.predict(X_test_full)

    # ── 5. Uncertainty ─────────────────────────────────────────────────────
    print("[5/6] Computing uncertainty...")
    confidence, uncertain_flag = compute_uncertainty(
        state_proba,
        text_lengths=test_df["journal_text"].fillna("").apply(len).values,
        reflection_quality=test_df.get(
            "reflection_quality", pd.Series(np.ones(len(test_df)))
        ).map({"conflicted": 0, "vague": 1, "clear": 2}).fillna(1).values
    )

    # ── 6. Decision engine ─────────────────────────────────────────────────
    print("[6/6] Running decision engine...")
    what_to_do, when_to_do = decide(
        predicted_state=predicted_state,
        predicted_intensity=predicted_intensity,
        stress_level=test_df["stress_level"].values,
        energy_level=test_df["energy_level"].values,
        time_of_day=test_df["time_of_day"].values,
        sleep_hours=test_df["sleep_hours"].values,
    )

    # ── Error analysis on train set ────────────────────────────────────────
    print("      Running error analysis on training set...")
    from src.error_analysis import generate_error_analysis
    train_pred_state, _ = state_model.predict_and_proba(X_train_full)
    train_pred_intensity = intensity_model.predict(X_train_full)
    train_conf, train_unc = compute_uncertainty(
        _,
        text_lengths=train_df["journal_text"].fillna("").apply(len).values,
    )
    generate_error_analysis(
        df=train_df,
        predicted_state=train_pred_state,
        true_state=train_df["emotional_state"].values,
        predicted_intensity=train_pred_intensity,
        true_intensity=train_df["intensity"].values,
        confidence=train_conf,
    )

    # ── Output ─────────────────────────────────────────────────────────────
    results = pd.DataFrame({
        "id":                  test_df["id"].values,
        "predicted_state":     predicted_state,
        "predicted_intensity": np.round(predicted_intensity, 2),
        "confidence":          np.round(confidence, 3),
        "uncertain_flag":      uncertain_flag,
        "what_to_do":          what_to_do,
        "when_to_do":          when_to_do,
    })
    save_predictions(results, out_path)

    print(f"\n✓ Saved {len(results)} predictions → {out_path}")
    print("\n── Distribution summary ──────────────────")
    print(results["predicted_state"].value_counts().to_string())
    print(f"\nUncertain rows: {uncertain_flag.sum()} / {len(uncertain_flag)}")
    print(f"\n✓ ERROR_ANALYSIS.md saved")
    print("==========================================\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    default="data/train.csv")
    parser.add_argument("--test",     default="data/test.csv")
    parser.add_argument("--out",      default="predictions.csv")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.train, args.test, args.out, ablation=args.ablation)