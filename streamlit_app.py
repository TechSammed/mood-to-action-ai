"""
streamlit_app.py — ArvyaX Mental State UI
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="ArvyaX · Mental State Guide",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size:2.2rem; font-weight:700;
    background:linear-gradient(135deg,#2ecc71,#3498db);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.state-card {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid #0f3460; border-radius:16px;
    padding:24px; text-align:center; color:white; margin-bottom:1rem;
}
.state-value { font-size:2.4rem; font-weight:700; color:#2ecc71; }
.state-label { font-size:0.82rem; color:#aaa; text-transform:uppercase; letter-spacing:1px; }
.decision-box {
    background:linear-gradient(135deg,#667eea,#764ba2);
    border-radius:14px; padding:20px; color:white; margin:1rem 0;
}
.message-box {
    background:#e8f5e9; border-left:4px solid #2ecc71;
    border-radius:8px; padding:16px; margin:1rem 0;
    font-style:italic; color:#1b5e20;
}
.uncertain-box {
    background:#fff3e0; border-left:4px solid #ff9800;
    border-radius:8px; padding:12px; margin:1rem 0; color:#e65100;
}
.confident-box {
    background:#e8f5e9; border-left:4px solid #4caf50;
    border-radius:8px; padding:12px; margin:1rem 0; color:#1b5e20;
}
</style>
""", unsafe_allow_html=True)

# ── Emoji maps ─────────────────────────────────────────────────────────────
STATE_EMOJI    = {"calm":"😌","focused":"🎯","neutral":"😐","restless":"🌀","overwhelmed":"🌊","mixed":"🔀"}
ACTIVITY_EMOJI = {"box_breathing":"🫁","journaling":"📔","grounding":"🌱","deep_work":"💻","yoga":"🧘","sound_therapy":"🎵","light_planning":"📋","rest":"🛌","movement":"🚶","pause":"⏸️"}
WHEN_EMOJI     = {"now":"⚡","within_15_min":"🕐","later_today":"🌤️","tonight":"🌙","tomorrow_morning":"🌅"}

# ── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        from src.models import load_models
        state_model, intensity_model = load_models(tag="full")
        text_vectorizer = joblib.load("models/text_vectorizer.pkl")
        meta_cols       = joblib.load("models/meta_cols.pkl")
        scaler          = joblib.load("models/scaler.pkl")
        return state_model, intensity_model, text_vectorizer, meta_cols, scaler, True
    except Exception as e:
        return None, None, None, None, None, False

def run_inference(inputs: dict):
    from src.features import build_features
    from src.decision import decide, generate_supportive_message
    from src.uncertainty import compute_uncertainty

    state_model, intensity_model, text_vectorizer, meta_cols, scaler, loaded = load_pipeline()
    if not loaded:
        return None

    df = pd.DataFrame([inputs])
    X_full, _, _, _, _ = build_features(df, fit=False, text_vectorizer=text_vectorizer, meta_cols=meta_cols, scaler=scaler)
    pred_state, proba    = state_model.predict_and_proba(X_full)
    pred_intensity       = intensity_model.predict(X_full)
    confidence, unc_flag = compute_uncertainty(proba, text_lengths=np.array([len(inputs["journal_text"])]))
    what_list, when_list = decide(
        pred_state, pred_intensity,
        np.array([inputs["stress_level"]]), np.array([inputs["energy_level"]]),
        np.array([inputs["time_of_day"]]),  np.array([inputs["sleep_hours"]]),
    )
    message = generate_supportive_message(
        pred_state[0], pred_intensity[0], what_list[0], when_list[0], float(confidence[0])
    )
    return {
        "predicted_state": pred_state[0], "predicted_intensity": int(pred_intensity[0]),
        "confidence": round(float(confidence[0]), 3), "uncertain_flag": int(unc_flag[0]),
        "what_to_do": what_list[0], "when_to_do": when_list[0],
        "message": message,
        "prob_dict": dict(zip(state_model.classes_, proba[0])),
    }

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 ArvyaX")
    st.markdown("*Mental State Intelligence*")
    st.divider()
    _, _, _, _, _, models_loaded = load_pipeline()
    if models_loaded:
        st.success("✅ Models loaded")
    else:
        st.error("⚠️ Models not found")
        st.info("Run `python main.py` first, then restart.")
    st.divider()
    page = st.radio("", ["🔍 Analyse My State", "📊 Batch Predictions", "📈 Model Insights", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.caption("ArvyaX × RevoltronX · 2024")


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSE
# ════════════════════════════════════════════════════════════════════
if page == "🔍 Analyse My State":
    st.markdown('<div class="main-title">🌿 How are you feeling?</div>', unsafe_allow_html=True)
    st.markdown("Write a short reflection after your session.")

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        journal_text = st.text_area(
            "Reflection",
            placeholder="e.g. The rain session was soothing but I can't stop thinking about work...",
            height=130, label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        with c1: ambience  = st.selectbox("Ambience", ["forest","ocean","rain","mountain","cafe"])
        with c2: duration  = st.slider("Duration (min)", 3, 120, 20)

        c1, c2, c3 = st.columns(3)
        with c1: sleep_hours  = st.slider("Sleep (hrs)", 3.0, 10.0, 7.0, 0.5)
        with c2: energy_level = st.slider("Energy (1–5)", 1, 5, 3)
        with c3: stress_level = st.slider("Stress (1–5)", 1, 5, 3)

        c1, c2, c3 = st.columns(3)
        with c1: time_of_day = st.selectbox("Time", ["early_morning","morning","afternoon","evening","night"])
        with c2: prev_mood   = st.selectbox("Yesterday mood", ["overwhelmed","restless","mixed","neutral","calm","focused"], index=3)
        with c3: face_hint   = st.selectbox("Face hint", ["tired_face","tense_face","neutral_face","calm_face","happy_face"], index=2)

        reflection_quality = st.select_slider("Reflection clarity", ["conflicted","vague","clear"], value="vague")
        analyse_btn = st.button("🔍 Analyse my state", type="primary", use_container_width=True)

    with col_right:
        if analyse_btn:
            if not journal_text.strip():
                st.warning("Please write something first.")
            else:
                with st.spinner("Reading your state..."):
                    result = run_inference({
                        "id": 0, "journal_text": journal_text,
                        "ambience_type": ambience, "duration_min": duration,
                        "sleep_hours": sleep_hours, "energy_level": energy_level,
                        "stress_level": stress_level, "time_of_day": time_of_day,
                        "previous_day_mood": prev_mood, "face_emotion_hint": face_hint,
                        "reflection_quality": reflection_quality,
                    })
                if result is None:
                    st.error("Models not loaded. Run python main.py first.")
                else:
                    s = result["predicted_state"]
                    emoji = STATE_EMOJI.get(s, "🧠")
                    stars = "★" * result["predicted_intensity"] + "☆" * (5 - result["predicted_intensity"])

                    st.markdown(f"""
                    <div class="state-card">
                        <div class="state-label">Detected emotional state</div>
                        <div class="state-value">{emoji} {s.title()}</div>
                        <div class="state-label">Intensity {result['predicted_intensity']} / 5 &nbsp;·&nbsp; {stars}</div>
                    </div>""", unsafe_allow_html=True)

                    conf = result["confidence"]
                    if result["uncertain_flag"]:
                        st.markdown(f'<div class="uncertain-box">⚠️ Low confidence ({conf:.0%}) — trust your gut too.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="confident-box">✅ Confidence: {conf:.0%}</div>', unsafe_allow_html=True)

                    w, wh = result["what_to_do"], result["when_to_do"]
                    st.markdown(f"""
                    <div class="decision-box">
                        <div style="font-size:0.8rem;opacity:0.8;text-transform:uppercase;letter-spacing:1px">Recommended action</div>
                        <div style="font-size:1.8rem;font-weight:700;margin:6px 0">{ACTIVITY_EMOJI.get(w,"🎯")} {w.replace("_"," ").title()}</div>
                        <div style="font-size:0.9rem;opacity:0.85">{WHEN_EMOJI.get(wh,"⏱️")} {wh.replace("_"," ").title()}</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown(f'<div class="message-box">💬 {result["message"]}</div>', unsafe_allow_html=True)

                    st.markdown("#### State probabilities")
                    prob_df = pd.DataFrame(list(result["prob_dict"].items()), columns=["State","Probability"]).sort_values("Probability")
                    fig = px.bar(prob_df, x="Probability", y="State", orientation="h",
                                 color="Probability", color_continuous_scale="Viridis", height=260)
                    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False,
                                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Fill in your reflection and click **Analyse my state**")


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Predictions":
    st.markdown('<div class="main-title">📊 Batch Predictions</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload test file (.xlsx or .csv)", type=["xlsx","csv"])

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        st.success(f"Loaded {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("▶️ Run predictions", type="primary"):
            from src.features import build_features
            from src.decision import decide
            from src.uncertainty import compute_uncertainty

            state_model, intensity_model, text_vectorizer, meta_cols, scaler, loaded = load_pipeline()
            if not loaded:
                st.error("Models not loaded.")
            else:
                with st.spinner(f"Running on {len(df)} rows..."):
                    df["journal_text"] = df["journal_text"].fillna("").astype(str)
                    X_full, _, _, _, _ = build_features(df, fit=False, text_vectorizer=text_vectorizer, meta_cols=meta_cols, scaler=scaler)
                    pred_state, proba    = state_model.predict_and_proba(X_full)
                    pred_intensity       = intensity_model.predict(X_full)
                    confidence, unc_flag = compute_uncertainty(proba, text_lengths=df["journal_text"].apply(len).values)
                    what_list, when_list = decide(
                        pred_state, pred_intensity,
                        df.get("stress_level", pd.Series(np.ones(len(df))*3)).values,
                        df.get("energy_level", pd.Series(np.ones(len(df))*3)).values,
                        df.get("time_of_day",  pd.Series(["morning"]*len(df))).values,
                        df.get("sleep_hours",  pd.Series(np.ones(len(df))*7)).values,
                    )

                results = pd.DataFrame({
                    "id": df.get("id", pd.RangeIndex(len(df))).values,
                    "predicted_state": pred_state,
                    "predicted_intensity": pred_intensity.astype(int),
                    "confidence": np.round(confidence, 3),
                    "uncertain_flag": unc_flag,
                    "what_to_do": what_list,
                    "when_to_do": when_list,
                })
                st.success("✅ Done!")
                st.dataframe(results, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(results, names="predicted_state", hole=0.4, height=300, title="State distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.bar(results["what_to_do"].value_counts().reset_index(), x="what_to_do", y="count", height=300, title="Activities")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button("⬇️ Download predictions.csv", results.to_csv(index=False).encode(), "predictions.csv", "text/csv", use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — INSIGHTS
# ════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.markdown('<div class="main-title">📈 Model Insights</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Decision logic", "Uncertainty", "Architecture"])

    with tab1:
        st.markdown("### Decision engine — state → activity")
        st.markdown("**Rule-based, not ML.** Transparent, safe, auditable.")
        map_df = pd.DataFrame([
            {"State": "calm",       "Emoji": "😌", "Activity (high energy)": "deep_work",    "Activity (low energy)": "light_planning"},
            {"State": "focused",    "Emoji": "🎯", "Activity (high energy)": "deep_work",    "Activity (low energy)": "light_planning"},
            {"State": "neutral",    "Emoji": "😐", "Activity (high energy)": "light_planning","Activity (low energy)": "pause"},
            {"State": "restless",   "Emoji": "🌀", "Activity (high energy)": "movement",     "Activity (low energy)": "yoga"},
            {"State": "overwhelmed","Emoji": "🌊", "Activity (high energy)": "box_breathing", "Activity (low energy)": "rest"},
            {"State": "mixed",      "Emoji": "🔀", "Activity (high energy)": "journaling",   "Activity (low energy)": "sound_therapy"},
        ])
        st.dataframe(map_df, use_container_width=True, hide_index=True)
        st.error("🔴 Safety: sleep < 5 + night/evening + energy ≤ 2 → always **rest**")
        st.error("🔴 Safety: stress ≥ 4 + overwhelmed + intensity ≥ 4 → always **box_breathing**")

    with tab2:
        st.markdown("### Confidence formula")
        st.latex(r"\text{confidence} = (0.50 \cdot p_{max} + 0.25 \cdot (1-H) + 0.25 \cdot \text{margin}) \times Q")
        st.markdown("Threshold: **0.45** — below this → `uncertain_flag = 1`")
        demo = st.slider("Demo confidence", 0.0, 1.0, 0.62)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=demo * 100,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0,100]}, "bar": {"color": "#2ecc71" if demo >= 0.45 else "#e67e22"},
                   "steps": [{"range":[0,45],"color":"#ffe0cc"},{"range":[45,100],"color":"#ccffe0"}],
                   "threshold": {"line": {"color":"red","width":3}, "value":45}},
            title={"text": "Uncertain if below red line"},
        ))
        fig.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Pipeline")
        st.code("""
Journal text                    Metadata (sleep, stress, energy…)
     │                                       │
     ▼                                       ▼
all-MiniLM-L6-v2            Ordinal encode + StandardScaler
(384-dim SBERT)              + derived features (13-dim)
+ text stats (7-dim)
     │                                       │
     └──────────── concat ──────────────────┘
                        │
               XGBoost (full features)
               /                   \\
 CalibratedClassifierCV       XGBRegressor
 (emotional_state)            (intensity 1–5)
         │                          │
    proba vector              clipped float
         │                          │
         └────── Uncertainty ───────┘
                (entropy + margin + quality)
                        │
                Decision Engine (rules)
                (what_to_do + when_to_do)
                        │
                 predictions.csv
        """, language="text")
        c1, c2, c3 = st.columns(3)
        c1.metric("SBERT encoder", "~22 MB", "INT8 quantised: ~6 MB")
        c2.metric("State model",    "~2 MB",  "XGBoost .pkl")
        c3.metric("Intensity model","~1 MB",  "XGBoost .pkl")


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="main-title">ℹ️ About ArvyaX</div>', unsafe_allow_html=True)
    st.markdown("""
## What this project does

ArvyaX is a **mental state guidance system** — not just a classifier.

| Step | What it does |
|---|---|
| **Understand** | Reads journal reflection + physiological context → detects emotional state + intensity |
| **Decide** | Picks the most helpful activity (breathing, rest, journaling, deep work…) and when |
| **Guide** | Generates a human-like supportive message |
| **Acknowledge uncertainty** | Flags low-confidence predictions instead of blindly guessing |

## Dataset
- **Train**: 1080 rows · 6 emotional states: `calm, focused, neutral, restless, overwhelmed, mixed`
- **Test**: 120 rows · IDs 10001–10120 · no labels (pure prediction task)
- Lots of short texts: *"bit restless"*, *"mind racing"*, *"still off"* — real-world noisiness

## Models
| Model | Task | Why |
|---|---|---|
| XGBoost + CalibratedClassifierCV | Emotional state | Calibrated probabilities for confidence scoring |
| XGBoost Regressor | Intensity 1–5 | Ordinal — regression preserves ordering |

## Tech stack
`sentence-transformers` · `xgboost` · `scikit-learn` · `FastAPI` · `Streamlit` · `Plotly`
    """)