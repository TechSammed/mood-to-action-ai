# ArvyaX Mental State Pipeline

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download SBERT model (once)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 3. Train + generate predictions.csv
python main.py --train data/train.csv --test data/test.csv --out predictions.csv

# 4. Launch Streamlit UI
streamlit run streamlit_app.py

# 5. Launch FastAPI (separate terminal)
uvicorn api:app --reload --port 8000
```

## Dataset columns
- `emotional_state`: calm, focused, neutral, restless, overwhelmed, mixed
- `reflection_quality`: clear, vague, conflicted
- `time_of_day`: early_morning, morning, afternoon, evening, night
- `energy_level` / `stress_level`: scale 1–5

## Outputs
| File | Description |
|---|---|
| `predictions.csv` | id, predicted_state, predicted_intensity, confidence, uncertain_flag, what_to_do, when_to_do |
| `ERROR_ANALYSIS.md` | 10+ failure cases with taxonomy |
| `reports/ablation_report.md` | Text-only vs full model comparison |