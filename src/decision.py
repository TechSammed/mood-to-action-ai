"""
src/decision.py
---------------
Rule-based decision engine using REAL dataset values.

Real emotional_state classes: calm, focused, neutral, restless, overwhelmed, mixed
Real time_of_day values: early_morning, morning, afternoon, evening, night
Real stress/energy scale: 1–5
"""

import numpy as np
from typing import List, Tuple

# ── State → base activity (6 real classes) ────────────────────────────────
STATE_BASE: dict = {
    "calm":       {"high": "deep_work",      "low": "light_planning", "default": "deep_work"},
    "focused":    {"high": "deep_work",      "low": "light_planning", "default": "deep_work"},
    "neutral":    {"high": "light_planning", "low": "pause",          "default": "light_planning"},
    "restless":   {"high": "movement",       "low": "yoga",           "default": "movement"},
    "overwhelmed":{"high": "box_breathing",  "low": "rest",           "default": "box_breathing"},
    "mixed":      {"high": "journaling",     "low": "sound_therapy",  "default": "journaling"},
}
_DEFAULT_STATE = {"high": "pause", "low": "rest", "default": "pause"}

# ── Time gating — includes early_morning ──────────────────────────────────
TIME_GATE: dict = {
    "deep_work":     ["morning", "early_morning", "afternoon"],
    "light_planning":["morning", "early_morning", "afternoon", "evening"],
    "movement":      ["morning", "early_morning", "afternoon", "evening"],
    "yoga":          ["morning", "early_morning", "evening"],
    "journaling":    ["morning", "evening", "night"],
    "box_breathing": ["morning", "early_morning", "afternoon", "evening", "night"],
    "grounding":     ["morning", "early_morning", "afternoon", "evening", "night"],
    "rest":          ["afternoon", "evening", "night"],
    "sound_therapy": ["morning", "early_morning", "afternoon", "evening", "night"],
    "pause":         ["morning", "early_morning", "afternoon", "evening", "night"],
}
_FALLBACK_ACTIVITY = {
    "morning":       "box_breathing",
    "early_morning": "box_breathing",
    "afternoon":     "pause",
    "evening":       "journaling",
    "night":         "rest",
}


def _when(state, intensity, stress, energy, time, sleep) -> str:
    s = state.lower()
    t = time.lower() if isinstance(time, str) else "morning"

    # Critical → act now (stress/energy on 1–5 scale)
    if s in ("overwhelmed", "restless") and intensity >= 4:
        return "now"
    if stress >= 4 and intensity >= 4:
        return "now"

    # Low energy at night
    if t in ("evening", "night") and energy <= 2:
        return "tonight"

    # Positive state, mild intensity → schedule soon
    if s in ("calm", "focused") and intensity <= 3:
        if t in ("morning", "early_morning", "afternoon"):
            return "within_15_min"
        return "later_today"

    if intensity >= 3:
        return "within_15_min"

    if t == "night":
        return "tonight"
    if t in ("morning", "early_morning") and sleep < 6:
        return "now"
    if t == "evening":
        return "later_today"

    return "within_15_min"


def _what(state, intensity, stress, energy, time, sleep) -> str:
    s = state.lower()
    t = time.lower() if isinstance(time, str) else "morning"

    base = STATE_BASE.get(s, _DEFAULT_STATE)

    # Safety overrides
    if sleep < 5 and t in ("night", "evening") and energy <= 2:
        return "rest"
    if stress >= 4 and s == "overwhelmed" and intensity >= 4:
        return "box_breathing"

    # Midpoint of 1–5 scale = 3
    rec = base["high"] if energy >= 3 else base["low"]

    allowed_times = TIME_GATE.get(rec, list(TIME_GATE.keys()))
    if t not in allowed_times:
        rec = _FALLBACK_ACTIVITY.get(t, "pause")

    return rec


def decide(
    predicted_state: np.ndarray,
    predicted_intensity: np.ndarray,
    stress_level: np.ndarray,
    energy_level: np.ndarray,
    time_of_day: np.ndarray,
    sleep_hours: np.ndarray,
) -> Tuple[List[str], List[str]]:
    stress_level        = np.nan_to_num(stress_level.astype(float),        nan=3.0)
    energy_level        = np.nan_to_num(energy_level.astype(float),        nan=3.0)
    sleep_hours         = np.nan_to_num(sleep_hours.astype(float),         nan=7.0)
    predicted_intensity = np.nan_to_num(predicted_intensity.astype(float), nan=3.0)

    what_list, when_list = [], []
    for i in range(len(predicted_state)):
        s   = str(predicted_state[i])
        ing = float(predicted_intensity[i])
        st  = float(stress_level[i])
        en  = float(energy_level[i])
        tod = str(time_of_day[i]) if time_of_day[i] else "morning"
        slp = float(sleep_hours[i])
        what_list.append(_what(s, ing, st, en, tod, slp))
        when_list.append(_when(s, ing, st, en, tod, slp))

    return what_list, when_list


def generate_supportive_message(state: str, intensity: float, what: str, when: str, confidence: float) -> str:
    intensity = int(np.clip(round(float(intensity)), 1, 5))
    qualifier = {1: "a hint of", 2: "some", 3: "noticeably", 4: "quite", 5: "very intensely"}[intensity]

    activity_phrases = {
        "box_breathing":  "a short box breathing exercise",
        "journaling":     "a few minutes of free journaling",
        "grounding":      "a simple grounding exercise",
        "deep_work":      "a focused deep-work session",
        "yoga":           "gentle yoga or stretching",
        "sound_therapy":  "calming ambient sound",
        "light_planning": "light task planning",
        "rest":           "real rest — no screens, no obligations",
        "movement":       "a short walk or body movement",
        "pause":          "a mindful pause",
    }
    when_phrases = {
        "now":               "Right now is the right time.",
        "within_15_min":     "Try to start within the next 15 minutes.",
        "later_today":       "Aim to fit this in later today.",
        "tonight":           "Save this for tonight when you wind down.",
        "tomorrow_morning":  "A fresh start tomorrow morning would be ideal.",
    }
    msg_templates = {
        "calm":       f"You're {qualifier} calm — a good state to move into meaningful work.",
        "focused":    f"You're {qualifier} focused. Ideal time to go deep on something that matters.",
        "neutral":    f"You're in a fairly {qualifier} neutral state. A gentle start works well here.",
        "restless":   f"There's {qualifier} restless energy here. Let's give it somewhere useful to go.",
        "overwhelmed":f"Things feel {qualifier} overwhelming right now. One breath at a time.",
        "mixed":      f"Your state feels {qualifier} mixed — a bit of everything. That's okay.",
    }
    activity = activity_phrases.get(what, f"a {what} practice")
    timing   = when_phrases.get(when, "when you're ready.")
    opening  = msg_templates.get(state.lower(), f"You seem {qualifier} {state.lower()}.")
    uncertain_note = " (I'm less certain about this read — trust your gut too.)" if confidence < 0.4 else ""
    return f"{opening} Try {activity}. {timing}{uncertain_note}"