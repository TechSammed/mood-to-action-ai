"""src/utils.py — I/O helpers."""

import pandas as pd
import numpy as np
import os


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure required column
    if "journal_text" not in df.columns:
        raise ValueError("Missing required column: journal_text")

    # Clean text
    df["journal_text"] = df["journal_text"].fillna("").astype(str)

    # Clean face_emotion_hint — "none" string → NaN
    if "face_emotion_hint" in df.columns:
        df["face_emotion_hint"] = df["face_emotion_hint"].replace("none", np.nan)

    return df


def save_predictions(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved→{path}")