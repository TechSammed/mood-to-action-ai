# 🧠 Mental State Guidance System

> **From Understanding Humans → To Guiding Them**
>
> A production-grade ML pipeline that reads post-session reflections, detects emotional state, and recommends personalised actions — with built-in uncertainty awareness.

---

## 📌 Project Overview

This is **not a standard classification project**.

Most ML systems stop at predicting a label. This system goes three steps further:

| Layer | What it does |
|---|---|
| **Understand** | Reads a messy, short journal reflection + physiological context signals to detect the user's emotional state and intensity |
| **Decide** | Determines the most helpful activity (breathing, journaling, deep work, rest…) and the right timing (now, tonight, tomorrow) |
| **Guide** | Generates a human-like supportive message explaining the recommendation |
| **Acknowledge** | Flags low-confidence predictions honestly instead of guessing blindly |

Built for the **ArvyaX × RevoltronX Machine Learning Internship Assignment**.

---

## 🎯 Aim and Goal

### Aim
Build an end-to-end ML system that can understand a user's mental state from noisy, short reflective text combined with contextual signals, and guide them toward a better mental state through personalised action recommendations.

### Goals
- Predict **emotional state** (6 classes) from journal text + metadata
- Predict **intensity** (1–5) of that emotional state
- Make a **decision** on what activity and when
- Model **uncertainty** — know when the system doesn't know
- Handle **real-world noise**: short texts, missing values, conflicting signals
- Run **fully locally** — no external APIs, no cloud dependency

---

## 📊 Dataset

### Source
Provided by ArvyaX × RevoltronX as part of the internship assignment.

### Files
| File | Rows | Purpose |
|---|---|---|
| `data/train.csv` | 1200 | Training data with labels |
| `data/test.csv` | 120 | Test data — IDs 10001–10120, no labels |

### Columns

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `journal_text` | string | User's post-session reflection (free text, often noisy) |
| `ambience_type` | categorical | Session ambience: `forest`, `ocean`, `rain`, `mountain`, `cafe` |
| `duration_min` | float | Session duration in minutes (3–120) |
| `sleep_hours` | float | Hours slept last night (3.5–8.5) |
| `energy_level` | int | Self-reported energy on scale 1–5 |
| `stress_level` | int | Self-reported stress on scale 1–5 |
| `time_of_day` | categorical | `early_morning`, `morning`, `afternoon`, `evening`, `night` |
| `previous_day_mood` | categorical | Yesterday's mood: `calm`, `focused`, `neutral`, `mixed`, `restless`, `overwhelmed` |
| `face_emotion_hint` | categorical | Detected facial expression: `calm_face`, `neutral_face`, `tense_face`, `tired_face`, `happy_face` |
| `reflection_quality` | categorical | Quality of the journal entry: `clear`, `vague`, `conflicted` |
| `emotional_state` | categorical | **Target label** — `calm`, `focused`, `neutral`, `restless`, `overwhelmed`, `mixed` |
| `intensity` | int | **Target label** — intensity of state on scale 1–5 |

> **Note:** `emotional_state` and `intensity` are only in training data. Test data has all other columns.

### Real-world noise in dataset
The dataset contains exactly the kind of noise you'd find in production:
- Very short texts: `"bit restless"`, `"ok session"`, `"mind racing"`, `"still off"`
- Missing values in `sleep_hours`, `previous_day_mood`, `face_emotion_hint`
- `"none"` as a literal string in `face_emotion_hint` (not NaN)
- Contradictory signals: calm journal text with `stress_level=5`
- Vague/conflicted reflection quality reducing label reliability

---

## 🏗 System Architecture

```
Input
  journal_text (str) + metadata
  sleep / stress / energy / time_of_day / ambience / face_hint / mood
        |
        v
Feature Engineering                              404 dims total
  SBERT all-MiniLM-L6-v2 embeddings ..........  384 dims
  Handcrafted text stats .......................    7 dims
    (word count, char count, avg word length,
     exclamation count, question count,
     uppercase ratio, sentiment polarity)
  Metadata features (ordinal + scaled) ........   13 dims
    (sleep_hours, stress_level, energy_level,
     duration_min, time_of_day_enc,
     previous_day_mood_enc, face_emotion_hint_enc,
     reflection_quality_enc, ambience_enc,
     energy_stress_ratio, sleep_deficit,
     high_stress_flag, low_energy_flag)
        |
   +----+-----------+
   |                |
XGBoostClassifier   XGBoostRegressor
+ CalibratedCV      ordinal intensity
emotional_state     intensity (1-5)
6 classes           clip + round
F1: 0.594           MAE: 1.311
   |                |
   +----+-----------+
        |
   Uncertainty Module
   confidence = (0.60 * p_max_rescaled + 0.40 * margin) * Q
   uncertain_flag = 1 if confidence < 0.35
        |
   Rule-based Decision Engine
   what_to_do (activity) + when_to_do (timing)
   uses: state + intensity + stress + energy + time + sleep
        |
   Supportive Message Generator
   template-based, state-aware phrasing
        |
   predictions.csv  +  ERROR_ANALYSIS.md
```

---

## 🤖 Models Used

| Model | Task | Why this choice |
|---|---|---|
| `XGBoostClassifier` + `CalibratedClassifierCV` | Emotional state (6 classes) | Handles mixed text+tabular features; calibration gives reliable probabilities for confidence scoring |
| `XGBoostRegressor` | Intensity prediction (1–5) | Intensity is **ordinal** — 3 is closer to 4 than to 1. Regression preserves this ordering. A classifier penalises "predict 2, true 3" the same as "predict 1, true 5" — that's wrong. |

### Supporting components
| Component | Library | Purpose |
|---|---|---|
| `all-MiniLM-L6-v2` | `sentence-transformers` | Text embedding — 384-dim, 22MB, runs fully locally |
| `StandardScaler` | `scikit-learn` | Normalise metadata features |
| `TruncatedSVD` | `scikit-learn` | Fallback if sentence-transformers unavailable |
| `TextBlob` | `textblob` | Sentiment polarity as a handcrafted text feature |

---

## ⚙️ Decision Engine

The decision layer is **rule-based, not ML**. This is intentional:
- Transparent and auditable — every decision can be explained
- Safe by default — cannot recommend high-effort tasks to exhausted users
- No distribution shift risk — rules don't hallucinate

### Activity lookup (state → recommended activity)

| State | High energy (>=3) | Low energy (<3) |
|---|---|---|
| calm | deep_work | light_planning |
| focused | deep_work | light_planning |
| neutral | light_planning | pause |
| restless | movement | yoga |
| overwhelmed | box_breathing | rest |
| mixed | journaling | sound_therapy |

### Safety overrides (highest priority)
- `sleep < 5h` + `night/evening` + `energy <= 2` → **always rest**
- `stress >= 4` + `overwhelmed` + `intensity >= 4` → **always box_breathing**

### Timing options
`now` | `within_15_min` | `later_today` | `tonight` | `tomorrow_morning`

---

## 📂 Folder Structure

```
reflect-ai-engine/
│
├── main.py                    # Entry point — full pipeline
├── api.py                     # FastAPI inference server
├── streamlit_app.py           # Streamlit UI
├── requirements.txt           # All dependencies
├── README.md                  # This file
├── EDGE_PLAN.md               # Mobile/on-device deployment plan
├── ERROR_ANALYSIS.md          # Auto-generated failure analysis
│
├── src/                       # Core modules
│   ├── __init__.py
│   ├── features.py            # Feature engineering (SBERT + metadata)
│   ├── models.py              # XGBoost state + intensity models
│   ├── decision.py            # Rule-based decision engine
│   ├── uncertainty.py         # Confidence scoring + uncertain_flag
│   ├── ablation.py            # Ablation study (Parts 5 & 6)
│   ├── error_analysis.py      # Error analysis (Part 7)
│   └── utils.py               # Data loading + I/O helpers
│
├── data/                      # Dataset (not pushed to git)
│   ├── train.csv              # 1200 rows with labels
│   └── test.csv               # 120 rows without labels
│
├── models/                    # Saved model artefacts (auto-created)
│   ├── state_model_full.pkl
│   ├── intensity_model_full.pkl
│   ├── text_vectorizer.pkl
│   ├── meta_cols.pkl
│   └── scaler.pkl
│
├── reports/                   # Auto-generated reports
│   └── ablation_report.md
│
└── .streamlit/                # Streamlit theme config
    └── config.toml
```

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Step 1 — Create environment
```bash
# Using conda (recommended)
conda create --prefix ./arv_env python=3.10 -y
conda activate ./arv_env

# OR using venv
python -m venv arv_env
arv_env\Scripts\activate     # Windows
source arv_env/bin/activate  # Mac/Linux
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Place data files
```
data/train.csv   ← training dataset (1200 rows)
data/test.csv    ← test dataset (120 rows)
```

---

## ▶️ How to Run

### Train models + generate predictions
```bash
# First run — trains and saves models (~5 minutes)
python main.py --train data/train.csv --test data/test.csv --out predictions.csv

# Second run onwards — loads saved models (~15 seconds)
python main.py --train data/train.csv --test data/test.csv --out predictions.csv

# Force retrain
python main.py --train data/train.csv --test data/test.csv --retrain

# With ablation study
python main.py --train data/train.csv --test data/test.csv --ablation
```

### Launch Streamlit UI
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Launch FastAPI server
```bash
uvicorn api:app --reload --port 8000
# Swagger UI at http://localhost:8000/docs
```

---

## 📤 Output Files

### predictions.csv
Generated by `main.py` for all 120 test rows.

| Column | Type | Description |
|---|---|---|
| `id` | int | Test row ID (10001–10120) |
| `predicted_state` | string | Detected emotional state |
| `predicted_intensity` | int | Intensity 1–5 |
| `confidence` | float | Model confidence 0–1 |
| `uncertain_flag` | int | 1 if confidence < 0.35 |
| `what_to_do` | string | Recommended activity |
| `when_to_do` | string | Recommended timing |

### ERROR_ANALYSIS.md
Auto-generated failure analysis with:
- 15 most interesting failure cases
- Failure taxonomy (short text, conflicting signals, noisy labels, state mismatch, intensity mismatch)
- Systemic insights and improvement suggestions

### reports/ablation_report.md
Comparison of text-only vs full model vs metadata-only model for both state prediction and intensity prediction.

---

## 🔬 Feature Engineering Details

### Text features (391 dims)
1. **SBERT embeddings** (384-dim) — `all-MiniLM-L6-v2` encodes the journal text into a dense semantic vector. Captures meaning, sentiment, and context far better than TF-IDF.
2. **Handcrafted stats** (7-dim) — word count, character count, average word length, exclamation count, question count, uppercase ratio, TextBlob sentiment polarity.

### Metadata features (13 dims)
Ordinal-encoded and scaled using StandardScaler:
- `time_of_day` → 0 (early_morning) to 4 (night)
- `previous_day_mood` → 0 (overwhelmed) to 5 (focused)
- `reflection_quality` → 0 (conflicted), 1 (vague), 2 (clear)
- `face_emotion_hint` → 0 (tired_face) to 4 (happy_face)
- `ambience_type` → 0 (forest) to 4 (cafe)
- Numeric: `sleep_hours`, `energy_level`, `stress_level`, `duration_min` (scaled)
- Derived: `energy_stress_ratio`, `sleep_deficit`, `high_stress_flag`, `low_energy_flag`

---

## 📉 Model Performance

| Metric | Value | Notes |
|---|---|---|
| State F1-weighted | 0.594 | 5-fold CV, 6 classes, noisy data |
| Intensity MAE | 1.311 | 5-fold CV, scale 1–5 |
| Uncertain rows (test) | ~74% | Expected — dataset has many short vague texts |

**Why F1=0.594 is reasonable:**
- 6 classes with significant overlap in vocabulary
- Many short ambiguous texts like "ok session", "bit restless"
- `mixed` class is inherently ambiguous — its texts overlap with all other classes
- The uncertainty system correctly flags these borderline cases

---

## 🔍 Uncertainty Modeling

```
confidence = (0.60 × p_max_rescaled + 0.40 × margin) × Q

where:
  p_max_rescaled = (max_prob - 0.167) / (0.80 - 0.167)
  margin         = gap between top-2 probabilities (rescaled)
  Q              = text_quality_penalty × reflection_quality_score

uncertain_flag = 1 if confidence < 0.35
```

**Why rescale max_prob?**
With 6 classes, random guessing gives max_prob = 1/6 = 0.167. A max_prob of 0.327 is actually 2× better than random, but the raw value looks low. Rescaling from [0.167, 0.80] to [0, 1] gives a realistic confidence score.

---

## 📱 Edge / On-Device Plan

See `EDGE_PLAN.md` for full details. Summary:

| Component | Size | Method |
|---|---|---|
| SBERT encoder | 6 MB | ONNX + INT8 quantisation |
| State model | ~2 MB | XGBoost .ubj native |
| Intensity model | ~1 MB | XGBoost .ubj native |
| **Total** | **~9 MB** | |

Target latency: **~50ms** on mid-range Android.
All inference on-device — no user data ever leaves the device.

---

## 🛡 Robustness

| Scenario | Handling |
|---|---|
| Very short text ("ok", "fine") | Text length penalty reduces confidence → uncertain_flag=1 |
| Missing values | Median imputation for numeric, mode for categorical |
| `"none"` string in face_emotion_hint | Converted to NaN in `utils.py` |
| Conflicting signals (calm text + high stress) | Confidence drops due to feature contradiction |
| Unknown emotional state at inference | Falls back to `_DEFAULT_STATE` in decision engine |
| Unseen time_of_day value | Maps to `afternoon` (median) |

---

## 🧪 Ablation Study Results

Run with `--ablation` flag. Results saved to `reports/ablation_report.md`.

| Variant | State F1 | Intensity MAE |
|---|---|---|
| Full (text + metadata) | **0.594** (baseline) | **1.311** (baseline) |
| Text only | ~0.570 | ~1.380 |
| Metadata only | ~0.420 | ~1.250 |

**Key insight:** Text embeddings dominate state prediction. Metadata is more important for intensity prediction (physiological signals correlate more with intensity than state).

---

## 🔧 Tech Stack

| Layer | Library | Version |
|---|---|---|
| Text embedding | sentence-transformers | >=2.2.0 |
| ML models | xgboost | >=2.0.0 |
| Feature engineering | scikit-learn | >=1.3.0 |
| Data | pandas, numpy | latest |
| Uncertainty | scipy | >=1.10.0 |
| API | fastapi + uvicorn | >=0.110.0 |
| UI | streamlit | >=1.32.0 |
| Charts | plotly | >=5.18.0 |
| Sentiment | textblob | >=0.17.1 |

---

## 📋 Assignment Coverage

| Part | Description | File |
|---|---|---|
| Part 1 | Emotional state prediction | `src/models.py` → `StateModel` |
| Part 2 | Intensity prediction (regression) | `src/models.py` → `IntensityModel` |
| Part 3 | Decision engine (what + when) | `src/decision.py` |
| Part 4 | Uncertainty modeling | `src/uncertainty.py` |
| Part 5 | Feature understanding | `src/ablation.py` → Part C |
| Part 6 | Ablation study | `src/ablation.py` |
| Part 7 | Error analysis | `src/error_analysis.py` + `ERROR_ANALYSIS.md` |
| Part 8 | Edge / offline plan | `EDGE_PLAN.md` |
| Part 9 | Robustness | `src/utils.py` + `src/features.py` |
| Bonus | FastAPI + Streamlit UI | `api.py` + `streamlit_app.py` |

---

## 👨‍💻 Interview Preparation Notes

**Q: Why XGBoost over a neural network?**
XGBoost handles mixed tabular + text features naturally, trains fast, and gives reliable feature importance. A neural network would need more data and longer training to beat it here.

**Q: Why regression for intensity instead of classification?**
Intensity is ordinal. A classifier treats "predict 1, true 5" the same as "predict 1, true 2" — both are wrong by equal penalty. Regression naturally encodes that intensity=3 is closer to 4 than to 1.

**Q: Why rule-based decision engine?**
Transparency and safety. An ML decision layer could recommend "deep work" to someone with sleep=3, stress=5, energy=1. Rules prevent dangerous outputs and are auditable by product teams without retraining.

**Q: Why is F1=0.594 acceptable?**
The dataset has 6 semantically overlapping classes, many short ambiguous texts, and noisy labels. The uncertainty system correctly flags these borderline cases. In production, uncertain predictions would trigger a clarifying question to the user rather than a wrong recommendation.

**Q: How does the system handle short texts like "ok session"?**
The text length penalty in the uncertainty module drops the confidence multiplier to 0.55 for texts under 20 characters. This forces uncertain_flag=1, preventing the system from making confident wrong predictions based on minimal signal.