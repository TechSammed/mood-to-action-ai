"""
ArvyaX Mental State ML Pipeline
================================
Usage:
    # First run - trains and saves models (~5 min)
    python main.py --train data/train.csv --test data/test.csv --out predictions.csv

    # Second run onwards - loads saved models (~15 sec)
    python main.py --train data/train.csv --test data/test.csv --out predictions.csv

    # Force retrain
    python main.py --train data/train.csv --test data/test.csv --retrain
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
from src.utils import load_data


def models_saved() -> bool:
    files = [
        "models/state_model_full.pkl",
        "models/intensity_model_full.pkl",
        "models/text_vectorizer.pkl",
        "models/meta_cols.pkl",
        "models/scaler.pkl",
    ]
    return all(os.path.exists(f) for f in files)


def run_pipeline(train_path, test_path, out_path, ablation=False, retrain=False):
    print("\n=== ArvyaX Mental State Pipeline ===\n")

    print("[1/6] Loading data...")
    test_df  = load_data(test_path)
    train_df = load_data(train_path)
    print(f"      Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    if models_saved() and not retrain:
        print("[2/6] Loading saved features from disk...")
        text_vectorizer = joblib.load("models/text_vectorizer.pkl")
        meta_cols       = joblib.load("models/meta_cols.pkl")
        scaler          = joblib.load("models/scaler.pkl")
        print("[3/6] Loading saved models from disk...")
        state_model, intensity_model = load_models(tag="full")
        print("      Models loaded from disk - skipped retraining (use --retrain to force)")
    else:
        print("[2/6] Building features...")
        X_train_full, X_train_text, meta_cols, text_vectorizer, scaler = build_features(train_df, fit=True)
        y_state     = train_df["emotional_state"]
        y_intensity = train_df["intensity"]
        print(f"      Feature dim: {X_train_full.shape[1]}")

        print("[3/6] Training models...")
        state_model     = train_emotional_state(X_train_full, y_state)
        intensity_model = train_intensity(X_train_full, y_intensity)

        os.makedirs("models", exist_ok=True)
        joblib.dump(text_vectorizer, "models/text_vectorizer.pkl")
        joblib.dump(meta_cols,       "models/meta_cols.pkl")
        joblib.dump(scaler,          "models/scaler.pkl")
        print("      Models saved to disk")

        if ablation:
            from src.ablation import run_ablation
            run_ablation(X_train_full, X_train_text, y_state, y_intensity, meta_cols, X_train_text.shape[1])

    print("[4/6] Generating predictions...")
    X_test_full, _, _, _, _ = build_features(test_df, fit=False, text_vectorizer=text_vectorizer, meta_cols=meta_cols, scaler=scaler)
    predicted_state, state_proba = state_model.predict_and_proba(X_test_full)
    predicted_intensity          = intensity_model.predict(X_test_full)

    print("[5/6] Computing uncertainty...")
    confidence, uncertain_flag = compute_uncertainty(
        state_proba,
        text_lengths=test_df["journal_text"].fillna("").apply(len).values,
        reflection_quality=test_df.get("reflection_quality", pd.Series(np.ones(len(test_df)))).map({"conflicted": 0, "vague": 1, "clear": 2}).fillna(1).values
    )

    print("[6/6] Running decision engine...")
    what_to_do, when_to_do = decide(
        predicted_state=predicted_state,
        predicted_intensity=predicted_intensity,
        stress_level=test_df["stress_level"].values,
        energy_level=test_df["energy_level"].values,
        time_of_day=test_df["time_of_day"].values,
        sleep_hours=test_df["sleep_hours"].values,
    )

    print("      Running error analysis...")
    from src.error_analysis import generate_error_analysis
    X_train2, _, _, _, _ = build_features(train_df, fit=False, text_vectorizer=text_vectorizer, meta_cols=meta_cols, scaler=scaler)
    train_pred_state, train_proba = state_model.predict_and_proba(X_train2)
    train_pred_intensity          = intensity_model.predict(X_train2)
    train_conf, _                 = compute_uncertainty(train_proba, text_lengths=train_df["journal_text"].fillna("").apply(len).values)
    generate_error_analysis(
        df=train_df,
        predicted_state=train_pred_state,
        true_state=train_df["emotional_state"].values,
        predicted_intensity=train_pred_intensity,
        true_intensity=train_df["intensity"].values,
        confidence=train_conf,
    )

    results = pd.DataFrame({
        "id":                  test_df["id"].values,
        "predicted_state":     predicted_state,
        "predicted_intensity": np.round(predicted_intensity, 2),
        "confidence":          np.round(confidence, 3),
        "uncertain_flag":      uncertain_flag,
        "what_to_do":          what_to_do,
        "when_to_do":          when_to_do,
    })
    results.to_csv(out_path, index=False)

    print(f"\n Saved {len(results)} predictions to {out_path}")
    print("\n-- Distribution --")
    print(results["predicted_state"].value_counts().to_string())
    print(f"\nUncertain rows: {uncertain_flag.sum()} / {len(uncertain_flag)}")
    print("ERROR_ANALYSIS.md saved")
    print("==========================================\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    default="data/train.csv")
    parser.add_argument("--test",     default="data/test.csv")
    parser.add_argument("--out",      default="predictions.csv")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--retrain",  action="store_true", help="Force retrain even if models exist")
    args = parser.parse_args()
    run_pipeline(args.train, args.test, args.out, ablation=args.ablation, retrain=args.retrain)