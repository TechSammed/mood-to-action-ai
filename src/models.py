"""
src/models.py
-------------
XGBoost models for emotional state (classification) and intensity (regression).

emotional_state → XGBoostClassifier + CalibratedClassifierCV (isotonic)
intensity (1-5) → XGBoostRegressor  treated as ordinal regression
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import joblib
import os

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


class StateModel:
    def __init__(self, tag: str = "full"):
        self.tag = tag
        self.le  = LabelEncoder()
        self.model: Optional[CalibratedClassifierCV] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: pd.Series) -> "StateModel":
        y_enc = self.le.fit_transform(y.astype(str))
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)

        base = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.model.fit(X, y_enc)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(base, X, y_enc, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"  [StateModel-{self.tag}] CV F1-weighted: {scores.mean():.3f} ± {scores.std():.3f}")
        return self

    def predict_and_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        preds = self.le.inverse_transform(idx)
        return preds, proba

    def get_feature_importance(self) -> np.ndarray:
        try:
            importances = []
            for est in self.model.calibrated_classifiers_:
                importances.append(est.estimator.feature_importances_)
            return np.mean(importances, axis=0)
        except Exception:
            return np.array([])

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "StateModel":
        return joblib.load(path)


class IntensityModel:
    """
    Regression for intensity (1–5).
    Why regression: intensity is ordinal — 3 is closer to 4 than to 1.
    Classifier would penalise "predict 2, true 3" same as "predict 1, true 5" — wrong.
    """
    def __init__(self, tag: str = "full"):
        self.tag = tag
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(self, X: np.ndarray, y: pd.Series) -> "IntensityModel":
        y_vals = y.astype(float).clip(1, 5)

        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y_vals)

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            self.model, X, y_vals,
            cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        print(f"  [IntensityModel-{self.tag}] CV MAE: {-scores.mean():.3f} ± {scores.std():.3f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict(X)
        return np.clip(np.round(raw), 1, 5)

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "IntensityModel":
        return joblib.load(path)


def train_emotional_state(X: np.ndarray, y: pd.Series, tag: str = "full") -> StateModel:
    print(f"  Training StateModel [{tag}]  shape={X.shape}  classes={y.nunique()}")
    m = StateModel(tag=tag)
    m.fit(X, y)
    os.makedirs("models", exist_ok=True)
    m.save(f"models/state_model_{tag}.pkl")
    return m


def train_intensity(X: np.ndarray, y: pd.Series, tag: str = "full") -> IntensityModel:
    print(f"  Training IntensityModel [{tag}]  shape={X.shape}")
    m = IntensityModel(tag=tag)
    m.fit(X, y)
    os.makedirs("models", exist_ok=True)
    m.save(f"models/intensity_model_{tag}.pkl")
    return m


def load_models(tag: str = "full"):
    return (
        StateModel.load(f"models/state_model_{tag}.pkl"),
        IntensityModel.load(f"models/intensity_model_{tag}.pkl"),
    )