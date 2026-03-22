"""
Mental State Guidance System
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Mental State Guide",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — config.toml handles all colors/text
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1300px; }

.state-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    margin-bottom: 1rem;
}
.state-emoji { font-size: 3rem; display: block; margin-bottom: 8px; }
.state-name  { font-size: 1.8rem; font-weight: 800; letter-spacing: 3px; text-transform: uppercase; }
.state-meta  { font-size: 0.82rem; color: #8b949e; margin-top: 6px; }

.result-row {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #63f5c8;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 14px;
}
.result-icon  { font-size: 1.8rem; flex-shrink: 0; }
.result-label { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; }
.result-value { font-size: 1.1rem; font-weight: 700; color: #63f5c8; margin-top: 2px; }
.result-sub   { font-size: 0.8rem; color: #8b949e; margin-top: 2px; }

.msg-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #388bfd;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.92rem;
    line-height: 1.7;
    font-style: italic;
}

.section-tag {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #8b949e;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 5px;
    border-bottom: 1px solid #30363d;
}

.placeholder {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 16px;
    padding: 60px 30px;
    text-align: center;
    margin-top: 1rem;
}

.tech-pill {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.78rem;
    color: #8b949e;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
STATE_EMOJI    = {"calm":"😌","focused":"🎯","neutral":"😐","restless":"🌀","overwhelmed":"🌊","mixed":"🔀"}
STATE_COLOR    = {"calm":"#3fb950","focused":"#58a6ff","neutral":"#8b949e","restless":"#f0883e","overwhelmed":"#f85149","mixed":"#bc8cff"}
ACTIVITY_EMOJI = {"box_breathing":"🫁","journaling":"📔","grounding":"🌱","deep_work":"💻","yoga":"🧘","sound_therapy":"🎵","light_planning":"📋","rest":"🛌","movement":"🚶","pause":"⏸️"}
WHEN_EMOJI     = {"now":"⚡","within_15_min":"🕐","later_today":"🌤️","tonight":"🌙","tomorrow_morning":"🌅"}

# ── Pipeline cached permanently ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — first time only...")
def load_pipeline():
    try:
        from src.models import load_models
        state_model, intensity_model = load_models(tag="full")
        return {
            "state_model":     state_model,
            "intensity_model": intensity_model,
            "text_vectorizer": joblib.load("models/text_vectorizer.pkl"),
            "meta_cols":       joblib.load("models/meta_cols.pkl"),
            "scaler":          joblib.load("models/scaler.pkl"),
            "loaded":          True,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}


def run_inference(pipeline, inputs):
    from src.features import build_features
    from src.decision import decide, generate_supportive_message
    from src.uncertainty import compute_uncertainty
    df = pd.DataFrame([inputs])
    X, _, _, _, _ = build_features(
        df, fit=False,
        text_vectorizer=pipeline["text_vectorizer"],
        meta_cols=pipeline["meta_cols"],
        scaler=pipeline["scaler"],
    )
    pred_state, proba    = pipeline["state_model"].predict_and_proba(X)
    pred_intensity       = pipeline["intensity_model"].predict(X)
    rq_map               = {"conflicted": 0, "vague": 1, "clear": 2}
    confidence, unc_flag = compute_uncertainty(
        proba,
        text_lengths=np.array([len(inputs["journal_text"])]),
        reflection_quality=np.array([rq_map.get(inputs.get("reflection_quality","vague"), 1)])
    )
    what_list, when_list = decide(
        pred_state, pred_intensity,
        np.array([inputs["stress_level"]]),
        np.array([inputs["energy_level"]]),
        np.array([inputs["time_of_day"]]),
        np.array([inputs["sleep_hours"]]),
    )
    message = generate_supportive_message(
        pred_state[0], pred_intensity[0],
        what_list[0], when_list[0], float(confidence[0])
    )
    return {
        "predicted_state":     pred_state[0],
        "predicted_intensity": int(pred_intensity[0]),
        "confidence":          round(float(confidence[0]), 3),
        "uncertain_flag":      int(unc_flag[0]),
        "what_to_do":          what_list[0],
        "when_to_do":          when_list[0],
        "message":             message,
        "prob_dict":           dict(zip(pipeline["state_model"].classes_, proba[0])),
    }


pipeline = load_pipeline()

# ── Session state ──────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Analyse"

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 Mental State Guide")
    st.caption("Guidance System v1.0")
    st.divider()

    if pipeline.get("loaded"):
        st.success("✅ Models ready")
    else:
        st.error("❌ Models not found — run `python main.py` first")

    st.divider()
    st.subheader("Navigation")

    nav_pages = {
        "🔍 Analyse State": "Analyse",
        "📊 Batch Predict":  "Batch",
        "📈 Insights":       "Insights",
        "ℹ️ About":          "About",
    }

    for label, key in nav_pages.items():
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

    st.divider()
    st.caption("XGBoost + SBERT\nFully local · No API needed")

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════
if page == "Analyse":

    st.markdown("## 🧠 Mental State Analysis")
    st.caption("Write your post-session reflection and fill in context signals for a personalised recommendation.")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="section-tag">Session Reflection</div>', unsafe_allow_html=True)
        journal_text = st.text_area(
            "Reflection",
            placeholder="Describe how you felt during and after the session. More detail = better analysis...",
            height=130,
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-tag">Session Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: ambience = st.selectbox("Ambience", ["forest","ocean","rain","mountain","cafe"])
        with c2: duration = st.slider("Duration (min)", 3, 120, 20)

        st.markdown('<div class="section-tag">Context Signals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: sleep_hours  = st.slider("Sleep (hrs)",  3.0, 10.0, 7.0, 0.5)
        with c2: energy_level = st.slider("Energy (1-5)", 1, 5, 3)
        with c3: stress_level = st.slider("Stress (1-5)", 1, 5, 3)

        c1, c2, c3 = st.columns(3)
        with c1: time_of_day = st.selectbox("Time", ["early_morning","morning","afternoon","evening","night"])
        with c2: prev_mood   = st.selectbox("Yesterday", ["overwhelmed","restless","mixed","neutral","calm","focused"], index=3)
        with c3: face_hint   = st.selectbox("Expression", ["tired_face","tense_face","neutral_face","calm_face","happy_face"], index=2)

        reflection_quality = st.select_slider(
            "Reflection Clarity",
            options=["conflicted","vague","clear"],
            value="vague"
        )
        st.markdown("")
        go_btn = st.button("🔍 Analyse My State", use_container_width=True)

    with right:
        if not pipeline.get("loaded"):
            st.error("Models not loaded. Run `python main.py` first.")

        elif go_btn:
            if not journal_text.strip():
                st.warning("Please write your reflection first.")
            else:
                with st.spinner("Analysing..."):
                    R = run_inference(pipeline, {
                        "id": 0,
                        "journal_text":       journal_text,
                        "ambience_type":      ambience,
                        "duration_min":       duration,
                        "sleep_hours":        sleep_hours,
                        "energy_level":       energy_level,
                        "stress_level":       stress_level,
                        "time_of_day":        time_of_day,
                        "previous_day_mood":  prev_mood,
                        "face_emotion_hint":  face_hint,
                        "reflection_quality": reflection_quality,
                    })

                s      = R["predicted_state"]
                emoji  = STATE_EMOJI.get(s, "🧠")
                scolor = STATE_COLOR.get(s, "#63f5c8")
                stars  = "★" * R["predicted_intensity"] + "☆" * (5 - R["predicted_intensity"])
                conf   = R["confidence"]

                st.markdown(f"""
                <div class="state-card">
                    <span class="state-emoji">{emoji}</span>
                    <div class="state-name" style="color:{scolor};">{s.upper()}</div>
                    <div class="state-meta">Intensity {R['predicted_intensity']} / 5 &nbsp;·&nbsp; {stars}</div>
                </div>""", unsafe_allow_html=True)

                conf_pct = int(conf * 100)
                bar_col  = "#3fb950" if conf >= 0.35 else "#f0883e"
                conf_lbl = f"✓ Confidence: {conf:.0%}" if conf >= 0.35 else f"⚠ Low confidence: {conf:.0%} — trust your gut"
                st.markdown(f"""
                <div style="font-size:0.82rem;color:{bar_col};font-weight:600;margin:8px 0 4px 0;">{conf_lbl}</div>
                <div style="background:#21262d;border-radius:6px;height:8px;margin-bottom:14px;overflow:hidden;">
                    <div style="width:{conf_pct}%;height:100%;background:{bar_col};border-radius:6px;"></div>
                </div>""", unsafe_allow_html=True)

                w  = R["what_to_do"]
                wh = R["when_to_do"]
                st.markdown(f"""
                <div class="result-row">
                    <div class="result-icon">{ACTIVITY_EMOJI.get(w,"🎯")}</div>
                    <div>
                        <div class="result-label">Recommended action</div>
                        <div class="result-value">{w.replace("_"," ").title()}</div>
                        <div class="result-sub">{WHEN_EMOJI.get(wh,"⏱")} {wh.replace("_"," ").title()}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f'<div class="msg-card">💬 {R["message"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-tag">State probability breakdown</div>', unsafe_allow_html=True)
                prob_df = pd.DataFrame(
                    list(R["prob_dict"].items()),
                    columns=["State","Prob"]
                ).sort_values("Prob", ascending=True)

                bar_colors = [STATE_COLOR.get(st_, "#30363d") if st_ == s else "#30363d"
                              for st_ in prob_df["State"].tolist()]
                fig = go.Figure(go.Bar(
                    x=prob_df["Prob"].tolist(),
                    y=prob_df["State"].tolist(),
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"  {p:.1%}" for p in prob_df["Prob"].tolist()],
                    textposition="outside",
                    textfont=dict(color="#8b949e", size=11),
                ))
                fig.update_layout(
                    height=220, margin=dict(l=0,r=60,t=4,b=4),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                               range=[0, max(prob_df["Prob"].tolist()) * 1.4]),
                    yaxis=dict(showgrid=False, color="#8b949e", tickfont=dict(color="#8b949e")),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div class="placeholder">
                <div style="font-size:2.8rem;opacity:0.25;margin-bottom:12px;">🧠</div>
                <h3 style="font-size:1.05rem;margin-bottom:8px;">Ready to analyse</h3>
                <p style="color:#8b949e;font-size:0.88rem;line-height:1.6;margin:0;">
                    Write your reflection, set context signals,<br>
                    then click <strong style="color:#63f5c8;">Analyse My State</strong>
                </p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH
# ══════════════════════════════════════════════════════════════════════════
elif page == "Batch":

    st.markdown("## 📊 Batch Predictions")
    st.caption("Upload your test dataset to generate predictions for all rows at once.")

    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv","xlsx"])

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows",     len(df))
        c2.metric("Columns",        df.shape[1])
        c3.metric("Missing values", int(df.isnull().sum().sum()))

        with st.expander("Preview data"):
            st.dataframe(df.head(5), use_container_width=True)

        if st.button("▶ Run Batch Predictions", use_container_width=True):
            if not pipeline.get("loaded"):
                st.error("Models not loaded.")
            else:
                from src.features import build_features
                from src.decision import decide
                from src.uncertainty import compute_uncertainty

                prog = st.progress(0, text="Building features...")
                df["journal_text"] = df["journal_text"].fillna("").astype(str)
                X, _, _, _, _ = build_features(df, fit=False,
                    text_vectorizer=pipeline["text_vectorizer"],
                    meta_cols=pipeline["meta_cols"],
                    scaler=pipeline["scaler"],
                )
                prog.progress(40, text="Predicting states...")
                pred_state, proba    = pipeline["state_model"].predict_and_proba(X)
                pred_intensity       = pipeline["intensity_model"].predict(X)
                prog.progress(70, text="Computing uncertainty...")
                confidence, unc_flag = compute_uncertainty(proba, text_lengths=df["journal_text"].apply(len).values)
                what_list, when_list = decide(
                    pred_state, pred_intensity,
                    df.get("stress_level", pd.Series(np.ones(len(df))*3)).values,
                    df.get("energy_level", pd.Series(np.ones(len(df))*3)).values,
                    df.get("time_of_day",  pd.Series(["morning"]*len(df))).values,
                    df.get("sleep_hours",  pd.Series(np.ones(len(df))*7)).values,
                )
                prog.progress(100, text="Done!")

                results = pd.DataFrame({
                    "id":                  df.get("id", pd.RangeIndex(len(df))).values,
                    "predicted_state":     pred_state,
                    "predicted_intensity": pred_intensity.astype(int),
                    "confidence":          np.round(confidence, 3),
                    "uncertain_flag":      unc_flag,
                    "what_to_do":          what_list,
                    "when_to_do":          when_list,
                })

                st.success(f"Done! {len(results)} predictions generated.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",          len(results))
                c2.metric("Uncertain",      int(results["uncertain_flag"].sum()))
                c3.metric("Avg confidence", f"{results['confidence'].mean():.0%}")
                c4.metric("Top state",      results["predicted_state"].mode()[0])
                st.dataframe(results, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    state_counts = results["predicted_state"].value_counts()
                    fig = go.Figure(go.Pie(
                        labels=state_counts.index.tolist(),
                        values=state_counts.values.tolist(),
                        hole=0.5,
                        marker=dict(colors=[STATE_COLOR.get(s,"#8b949e") for s in state_counts.index.tolist()]),
                    ))
                    fig.update_layout(title="State distribution", height=300,
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#e6edf3"), margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    act_counts = results["what_to_do"].value_counts()
                    fig = go.Figure(go.Bar(
                        x=act_counts.values.tolist(),
                        y=act_counts.index.tolist(),
                        orientation="h",
                        marker=dict(color="#3fb950", line=dict(width=0)),
                        text=act_counts.values.tolist(),
                        textposition="outside",
                        textfont=dict(color="#8b949e"),
                    ))
                    fig.update_layout(
                        title="Recommended activities", height=300,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e6edf3"),
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        yaxis=dict(color="#8b949e"),
                        margin=dict(l=0,r=40,t=40,b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download predictions.csv", csv, "predictions.csv", "text/csv", use_container_width=True)
    else:
        st.info("Upload your test CSV or Excel file to get started.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Insights":

    st.markdown("## 📈 System Insights")
    st.caption("How the model works, how decisions are made, and how confidence is calculated.")

    tab1, tab2, tab3 = st.tabs(["Decision Logic", "Confidence System", "Architecture"])

    with tab1:
        st.markdown("### State to activity mapping")
        st.info("The decision engine is **rule-based, not ML** — transparent, safe, and auditable.")
        map_df = pd.DataFrame([
            {"State":"calm",        "High energy (>=3)":"deep_work",     "Low energy (<3)":"light_planning","Safety":"none"},
            {"State":"focused",     "High energy (>=3)":"deep_work",     "Low energy (<3)":"light_planning","Safety":"none"},
            {"State":"neutral",     "High energy (>=3)":"light_planning","Low energy (<3)":"pause",         "Safety":"none"},
            {"State":"restless",    "High energy (>=3)":"movement",      "Low energy (<3)":"yoga",          "Safety":"none"},
            {"State":"overwhelmed", "High energy (>=3)":"box_breathing", "Low energy (<3)":"rest",          "Safety":"stress>=4+intensity>=4 => breathing"},
            {"State":"mixed",       "High energy (>=3)":"journaling",    "Low energy (<3)":"sound_therapy", "Safety":"none"},
        ])
        st.dataframe(map_df, use_container_width=True, hide_index=True)
        st.markdown("#### Safety overrides")
        c1, c2 = st.columns(2)
        with c1: st.error("Sleep<5h + night/evening + energy<=2  =>  always REST")
        with c2: st.error("Stress>=4 + overwhelmed + intensity>=4  =>  always BOX BREATHING")

    with tab2:
        st.markdown("### Confidence formula")
        st.latex(r"\text{conf} = (0.60 \cdot \tilde{p}_{max} + 0.40 \cdot \text{margin}) \times Q")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
| Symbol | Meaning |
|---|---|
| Max prob rescaled | How decisive the model is |
| Margin | Gap between top-2 predictions |
| Q | Text length x reflection clarity |
| **Threshold** | **0.35 — below = uncertain** |
            """)
        with c2:
            demo = st.slider("Simulate confidence", 0.0, 1.0, 0.55, 0.01)
            col  = "#3fb950" if demo >= 0.35 else "#f0883e"
            fig  = go.Figure(go.Indicator(
                mode="gauge+number", value=round(demo*100, 1),
                number={"suffix":"%","font":{"size":32,"color":col}},
                gauge={
                    "axis":{"range":[0,100],"tickwidth":0},
                    "bar": {"color":col,"thickness":0.25},
                    "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                    "steps":[
                        {"range":[0,  35],"color":"rgba(248,81,73,0.08)"},
                        {"range":[35,100],"color":"rgba(63,185,80,0.08)"},
                    ],
                    "threshold":{"line":{"color":"#f85149","width":3},"value":35},
                },
                title={"text":"Red line = uncertain threshold","font":{"size":12,"color":"#8b949e"}},
            ))
            fig.update_layout(height=260, margin=dict(l=20,r=20,t=30,b=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Full pipeline")
        st.code("""
Input: journal_text + metadata
        |
Feature Engineering         404 dims
  SBERT embeddings ....  384
  Text stats ..........    7
  Metadata scaled .....   13
        |
   +----+----+
XGBoost    XGBoost
Classifier Regressor
state      intensity
F1: 0.594  MAE: 1.311
   +----+----+
        |
  Uncertainty (conf + flag)
        |
  Decision Engine (rules)
  what + when
        |
  Output + Message
        """, language="text")
        c1, c2, c3 = st.columns(3)
        c1.metric("SBERT encoder",   "22 MB",  "all-MiniLM-L6-v2")
        c2.metric("State model",     "~2 MB",  "XGBoost + calibration")
        c3.metric("Intensity model", "~1 MB",  "XGBoost regressor")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("## ℹ️ About This System")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### What it does")
        st.markdown("""
| Layer | Description |
|---|---|
| **Understand** | Detects emotional state + intensity from reflection + context |
| **Decide** | Picks best activity and timing |
| **Guide** | Generates a supportive message |
| **Uncertainty** | Flags low-confidence predictions honestly |
        """)
        st.markdown("### Dataset")
        st.markdown("""
- **Train** : 1200 rows · 6 states
- **States** : calm · focused · neutral · restless · overwhelmed · mixed
- **Test**  : 120 rows · IDs 10001-10120
        """)

    with c2:
        st.markdown("### Model choices")
        st.markdown("""
| Component | Choice | Why |
|---|---|---|
| Text | SBERT all-MiniLM-L6-v2 | Rich semantics, 22MB, local |
| State | XGBoost + Calibration | Reliable confidence |
| Intensity | XGBoost Regressor | Ordinal ordering |
| Decision | Rule-based | Transparent + safe |
        """)
        st.markdown("### Tech stack")
        techs = ["sentence-transformers","xgboost","scikit-learn","fastapi","streamlit","plotly","pandas","numpy"]
        cols  = st.columns(4)
        for i, t in enumerate(techs):
            cols[i%4].markdown(f'<div class="tech-pill">{t}</div>', unsafe_allow_html=True)

        st.markdown("### Performance")
        pc1, pc2 = st.columns(2)
        pc1.metric("State F1", "0.594", "6 classes")
        pc2.metric("Intensity MAE", "1.311", "1-5 scale")