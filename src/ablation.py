"""
src/ablation.py — Parts 5 & 6: Feature understanding + Ablation study
"""

import numpy as np
import pandas as pd
import os
from typing import List
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def run_ablation(
    X_full: np.ndarray,
    X_text: np.ndarray,
    y_state: pd.Series,
    y_intensity: pd.Series,
    meta_col_names: List[str],
    text_dim: int,
) -> str:
    lines = ["# Ablation Study Report\n\n"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_state.astype(str))
    X_meta = X_full[:, text_dim:]

    base_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )

    lines.append("## Emotional State — F1-weighted\n\n")
    lines.append("| Variant | CV F1 | Δ |\n|---|---|---|\n")

    full_f1  = cross_val_score(base_xgb, X_full,  y_enc, cv=cv, scoring="f1_weighted", n_jobs=-1).mean()
    text_f1  = cross_val_score(base_xgb, X_text,  y_enc, cv=cv, scoring="f1_weighted", n_jobs=-1).mean()
    meta_f1  = cross_val_score(base_xgb, X_meta,  y_enc, cv=cv, scoring="f1_weighted", n_jobs=-1).mean()

    lines.append(f"| Full (text + meta) | {full_f1:.3f} | baseline |\n")
    lines.append(f"| Text only          | {text_f1:.3f} | {text_f1 - full_f1:+.3f} |\n")
    lines.append(f"| Metadata only      | {meta_f1:.3f} | {meta_f1 - full_f1:+.3f} |\n\n")

    reg = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1
    )
    y_int = y_intensity.astype(float).clip(1, 5)

    lines.append("## Intensity — MAE\n\n")
    lines.append("| Variant | CV MAE | Δ |\n|---|---|---|\n")

    full_mae = -cross_val_score(reg, X_full,  y_int, cv=5, scoring="neg_mean_absolute_error").mean()
    text_mae = -cross_val_score(reg, X_text,  y_int, cv=5, scoring="neg_mean_absolute_error").mean()
    meta_mae = -cross_val_score(reg, X_meta,  y_int, cv=5, scoring="neg_mean_absolute_error").mean()

    lines.append(f"| Full (text + meta) | {full_mae:.3f} | baseline |\n")
    lines.append(f"| Text only          | {text_mae:.3f} | {text_mae - full_mae:+.3f} |\n")
    lines.append(f"| Metadata only      | {meta_mae:.3f} | {meta_mae - full_mae:+.3f} |\n\n")

    lines.append("## Feature Importance\n\n")
    base_xgb.fit(X_full, y_enc)
    importances = base_xgb.feature_importances_
    text_imp = importances[:text_dim].sum()
    meta_imp = importances[text_dim:]

    lines.append(f"| Group | Importance |\n|---|---|\n")
    lines.append(f"| Text (embeddings + stats) | {text_imp:.4f} |\n")
    for i, col in enumerate(meta_col_names):
        if i < len(meta_imp):
            lines.append(f"| {col} | {meta_imp[i]:.4f} |\n")

    report = "".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/ablation_report.md", "w") as f:
        f.write(report)
    print("  Ablation report → reports/ablation_report.md")
    return report