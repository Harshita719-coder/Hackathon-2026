import os
import time
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "buchwald_hartwig_huggingface.csv"

REAL_PARAMETERS = ["ligand", "additive", "base", "aryl_halide"]
SUCCESS_THRESHOLD = 80.0
MAX_ATTEMPTS = 3
NOISE_STD_PERCENT = 2.0


st.set_page_config(
    page_title="Cloud Lab Recovery Agent",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# UI CSS
# ============================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #020617 0%, #0f172a 40%, #0f766e 100%);
        color: white;
        margin-bottom: 1.3rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        opacity: 0.93;
        max-width: 1100px;
        line-height: 1.55;
    }

    .tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.45rem;
    }

    .info-box {
        background: #ffffff;
        border-left: 5px solid #0f766e;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .warning-box {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        border-top: 1px solid #fde68a;
        border-right: 1px solid #fde68a;
        border-bottom: 1px solid #fde68a;
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        height: 145px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 900;
        color: #0f172a;
        margin-top: 0.25rem;
    }

    .metric-note {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.35rem;
    }

    .pill-green {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .pill-blue {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .pill-red {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .pill-amber {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #fef3c7;
        color: #92400e;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .workflow-card {
        padding: 1rem;
        background: white;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        height: 100%;
    }

    .workflow-number {
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #0f766e;
        color: white;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }

    .workflow-title {
        font-size: 0.95rem;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .workflow-text {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.35;
    }

    .attempt-card {
        padding: 1rem;
        background: white;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.75rem;
    }

    .attempt-title {
        font-size: 1.05rem;
        font-weight: 900;
        color: #0f172a;
    }

    .small-muted {
        font-size: 0.85rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OPENAI DIAGNOSIS LAYER
# ============================================================
def get_openai_api_key():
    """
    Reads OpenAI API key safely.

    Priority:
    1. Streamlit secrets
    2. Environment variable OPENAI_API_KEY

    Do not hardcode keys in this file.
    """

    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY", None)
        if key_from_secrets:
            return key_from_secrets
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None

    api_key = get_openai_api_key()

    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def openai_diagnosis_agent(
    failure_type,
    protocol,
    yield_percent,
    top_parameter,
    ord_hint,
):
    """
    Uses OpenAI only for scientist-facing diagnosis text.
    The oracle and recovery loop remain local and deterministic.
    """

    client = get_openai_client()

    if client is None:
        return None

    prompt = f"""
You are helping a scientist use a cloud lab synthesis recovery agent.

Project:
Autonomous Synthesis Recovery Agent for Buchwald-Hartwig reactions.

Use only these real dataset parameters:
- ligand
- additive
- base
- aryl_halide

Do not mention temperature, time, solvent, catalyst loading, THF, Dioxane, HPLC purity,
or catalyst percent because those fields are not available in this dataset.

Current experiment:
- ligand: {protocol["ligand"]}
- additive: {protocol["additive"]}
- base: {protocol["base"]}
- aryl_halide: {protocol["aryl_halide"]}
- measured yield: {yield_percent}%
- failure type: {failure_type}
- strongest data lever: {top_parameter}
- local ORD-style hint: {ord_hint}

Write exactly 2 short sentences in simple English:
1. Explain what happened.
2. Explain what the cloud lab should test next.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content.strip()

        if text:
            return text

        return None

    except Exception:
        return None


def show_openai_status():
    api_key = get_openai_api_key()

    if OPENAI_AVAILABLE and api_key:
        st.success("OpenAI diagnosis: connected")
    elif not OPENAI_AVAILABLE:
        st.warning("OpenAI diagnosis: package not installed")
    else:
        st.warning("OpenAI diagnosis: local fallback")


# ============================================================
# FALLBACK DEMO DATA
# ============================================================
FALLBACK_HISTORY = [
    {
        "attempt": 1,
        "cloud_status": "Result received",
        "protocol": {
            "ligand": "L1",
            "additive": "A1",
            "base": "B1",
            "aryl_halide": "Selected substrate",
        },
        "yield_percent": 18.4,
        "true_yield_percent": 18.0,
        "failure_type": "low_yield",
        "diagnosis": (
            "Low yield detected. The agent should recover by changing ligand, additive, "
            "and base for the same aryl halide."
        ),
        "recovery_reason": (
            "Recovery Agent selected a stronger ligand/additive/base combination "
            "for the same aryl halide."
        ),
        "admet": "not_checked",
        "ord_hint": "Local successful reactions suggest checking the base choice.",
        "protocol_json": {},
    },
    {
        "attempt": 2,
        "cloud_status": "Result received",
        "protocol": {
            "ligand": "L3",
            "additive": "A8",
            "base": "B2",
            "aryl_halide": "Selected substrate",
        },
        "yield_percent": 54.2,
        "true_yield_percent": 53.7,
        "failure_type": "moderate_yield",
        "diagnosis": (
            "The reaction improved but did not reach the target. The agent should "
            "continue optimizing ligand, additive, and base choices."
        ),
        "recovery_reason": "Recovery Agent explored another untried high-confidence condition.",
        "admet": "not_checked",
        "ord_hint": "Local successful reactions support trying a different base.",
        "protocol_json": {},
    },
    {
        "attempt": 3,
        "cloud_status": "Target reached",
        "protocol": {
            "ligand": "L4",
            "additive": "A12",
            "base": "B2",
            "aryl_halide": "Selected substrate",
        },
        "yield_percent": 84.1,
        "true_yield_percent": 83.5,
        "failure_type": "success",
        "diagnosis": "The target yield was reached. The autonomous loop can stop.",
        "recovery_reason": "Success threshold reached.",
        "admet": "not_checked",
        "ord_hint": "Final condition reached the yield target.",
        "protocol_json": {},
    },
]

FALLBACK_IMPORTANCE = {
    "aryl_halide": 44.5,
    "additive": 31.2,
    "ligand": 24.7,
    "base": 11.9,
}


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(csv_path):
    path = Path(csv_path)

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = ["ligand", "additive", "base", "aryl_halide", "yield", "split"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_train_df(df):
    train_df = df[df["split"] == "train"].copy()

    if train_df.empty:
        raise ValueError("Train split is empty.")

    return train_df


# ============================================================
# CORE AUTONOMOUS LAB LOGIC
# ============================================================
class BHOracle:
    """
    Virtual lab oracle.

    In a real cloud lab, a robot/technician would run the protocol and return a yield.
    In this demo, the Buchwald-Hartwig CSV simulates the lab result.
    """

    def __init__(self, train_df, noise_std_percent=2.0, seed=42):
        self.train_df = train_df.copy()
        self.noise_std_percent = noise_std_percent
        self.rng = np.random.default_rng(seed)

    def query(self, protocol):
        exact = self.train_df[
            (self.train_df["ligand"] == protocol["ligand"])
            & (self.train_df["additive"] == protocol["additive"])
            & (self.train_df["base"] == protocol["base"])
            & (self.train_df["aryl_halide"] == protocol["aryl_halide"])
        ]

        if len(exact) > 0:
            true_yield = float(exact["yield"].mean())
            source = "exact_train_match"
            confidence = 0.95
            reaction_smiles = (
                exact["reaction_SMILES"].iloc[0]
                if "reaction_SMILES" in exact.columns
                else None
            )

        else:
            fallback = self.train_df[
                (self.train_df["base"] == protocol["base"])
                & (self.train_df["aryl_halide"] == protocol["aryl_halide"])
            ]

            if len(fallback) > 0:
                true_yield = float(fallback["yield"].mean())
                source = "fallback_same_base_and_aryl_halide"
                confidence = 0.65
                reaction_smiles = (
                    fallback["reaction_SMILES"].iloc[0]
                    if "reaction_SMILES" in fallback.columns
                    else None
                )

            else:
                true_yield = float(self.train_df["yield"].mean())
                source = "fallback_train_mean"
                confidence = 0.35
                reaction_smiles = None

        noisy_yield = true_yield + self.rng.normal(0, self.noise_std_percent)
        noisy_yield = float(np.clip(noisy_yield, 0, 100))

        return {
            "yield_percent": round(noisy_yield, 2),
            "true_yield_percent": round(true_yield, 2),
            "source": source,
            "confidence": confidence,
            "reaction_SMILES": reaction_smiles,
        }


class SimpleAnalyzer:
    """
    Lightweight local analyzer.

    Importance = how much average yield changes across each parameter.
    This is stable for live demos and avoids scipy/sklearn issues.
    """

    def __init__(self, train_df):
        self.train_df = train_df.copy()

    def importance(self):
        scores = {}

        for feature in REAL_PARAMETERS:
            grouped = self.train_df.groupby(feature)["yield"].mean()

            if len(grouped) <= 1:
                scores[feature] = 0.0
            else:
                scores[feature] = round(float(grouped.max() - grouped.min()), 2)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def predict_yield(self, protocol):
        estimates = []

        for feature in REAL_PARAMETERS:
            subset = self.train_df[self.train_df[feature] == protocol[feature]]

            if len(subset) > 0:
                estimates.append(float(subset["yield"].mean()))

        if estimates:
            return round(float(np.mean(estimates)), 2)

        return round(float(self.train_df["yield"].mean()), 2)


def classify_failure(
    yield_percent, previous_yield=None, confidence=0.95, success_threshold=80.0
):
    if confidence < 0.5:
        return "high_uncertainty"

    if previous_yield is not None:
        if yield_percent - previous_yield < 5 and yield_percent < success_threshold:
            return "no_improvement"

    if yield_percent < 30:
        return "low_yield"

    if yield_percent < 60:
        return "moderate_yield"

    if yield_percent < success_threshold:
        return "near_success"

    return "success"


def local_fallback_diagnosis(failure_type, top_parameter, ord_hint):
    if failure_type == "low_yield":
        message = (
            "Low yield detected. The cloud lab should change the available reaction "
            "choices instead of repeating the same failed condition."
        )
    elif failure_type == "moderate_yield":
        message = (
            "The reaction improved but is still below target. The agent should continue "
            "testing stronger ligand, additive, and base choices."
        )
    elif failure_type == "near_success":
        message = (
            "The reaction is close to success. The agent should make a smaller change "
            "toward a higher-yield condition."
        )
    elif failure_type == "no_improvement":
        message = (
            "The latest attempt did not improve enough. The cloud lab should explore "
            "a different untried condition."
        )
    elif failure_type == "success":
        message = "The yield target was reached. The cloud lab campaign can stop."
    else:
        message = "The agent selected the next valid recovery strategy."

    message += f" Current top data lever: {top_parameter}."

    if ord_hint:
        message += f" ORD-style local hint: {ord_hint}"

    return message


def make_diagnosis(
    failure_type, top_parameter, ord_hint, protocol=None, yield_percent=None
):
    if protocol is not None and yield_percent is not None:
        ai_text = openai_diagnosis_agent(
            failure_type=failure_type,
            protocol=protocol,
            yield_percent=yield_percent,
            top_parameter=top_parameter,
            ord_hint=ord_hint,
        )

        if ai_text:
            return ai_text

    return local_fallback_diagnosis(failure_type, top_parameter, ord_hint)


def local_ord_hint(train_df, aryl_halide):
    subset = train_df[
        (train_df["aryl_halide"] == aryl_halide) & (train_df["yield"] >= 70)
    ]

    if subset.empty:
        return "No high-yield local matches found for this aryl halide."

    counts = subset["base"].value_counts()
    base = counts.index[0]
    count = int(counts.iloc[0])

    return f"base {base} appears in {count} high-yield local reactions."


def find_starting_protocol(train_df, aryl_halide):
    subset = train_df[train_df["aryl_halide"] == aryl_halide].copy()
    low = subset[subset["yield"] < 30].copy()

    if not low.empty:
        row = low.sample(1, random_state=42).iloc[0]
    elif not subset.empty:
        row = subset.sort_values("yield", ascending=True).iloc[0]
    else:
        row = train_df.sort_values("yield", ascending=True).iloc[0]

    return {
        "ligand": row["ligand"],
        "additive": row["additive"],
        "base": row["base"],
        "aryl_halide": row["aryl_halide"],
    }


def build_cloud_lab_protocol(attempt, protocol):
    return {
        "campaign_id": f"BH-CLOUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "attempt": int(attempt),
        "objective": "maximize Buchwald-Hartwig reaction yield",
        "execution_layer": "virtual_cloud_lab_oracle",
        "measurement": "simulated_yield_percent",
        "reaction_condition": {
            "ligand": str(protocol["ligand"]),
            "additive": str(protocol["additive"]),
            "base": str(protocol["base"]),
            "aryl_halide": str(protocol["aryl_halide"]),
        },
        "safety_note": "This app uses historical reaction data as a virtual oracle. No physical synthesis is performed.",
    }


def recover_next_protocol(train_df, analyzer, current_protocol, tried_protocols):
    same_aryl = train_df[
        train_df["aryl_halide"] == current_protocol["aryl_halide"]
    ].copy()

    if same_aryl.empty:
        same_aryl = train_df.copy()

    tried = set()

    for p in tried_protocols:
        tried.add((p["ligand"], p["additive"], p["base"], p["aryl_halide"]))

    scored = []

    for _, row in same_aryl.iterrows():
        candidate = {
            "ligand": row["ligand"],
            "additive": row["additive"],
            "base": row["base"],
            "aryl_halide": row["aryl_halide"],
        }

        key = (
            candidate["ligand"],
            candidate["additive"],
            candidate["base"],
            candidate["aryl_halide"],
        )

        if key in tried:
            continue

        predicted = analyzer.predict_yield(candidate)

        scored.append(
            {
                "protocol": candidate,
                "predicted_yield": predicted,
            }
        )

    if not scored:
        return (
            current_protocol,
            "No untried condition found. Retrying current condition.",
        )

    scored = sorted(scored, key=lambda x: x["predicted_yield"], reverse=True)
    best = scored[0]["protocol"]

    changed = []

    for param in REAL_PARAMETERS:
        if best[param] != current_protocol[param]:
            changed.append(param)

    if changed:
        reason = "Recovery Agent changed " + ", ".join(changed)
    else:
        reason = "Recovery Agent kept the same condition"

    reason += " using the simple predicted-yield model."

    return best, reason


def run_live_campaign(train_df, selected_aryl_halide, success_threshold, max_attempts):
    oracle = BHOracle(train_df, noise_std_percent=NOISE_STD_PERCENT)
    analyzer = SimpleAnalyzer(train_df)
    importance = analyzer.importance()

    top_parameter = list(importance.keys())[0] if importance else "unknown"
    current_protocol = find_starting_protocol(train_df, selected_aryl_halide)

    history = []
    previous_yield = None

    for attempt in range(1, max_attempts + 1):
        protocol_json = build_cloud_lab_protocol(attempt, current_protocol)

        result = oracle.query(current_protocol)

        failure_type = classify_failure(
            result["yield_percent"],
            previous_yield=previous_yield,
            confidence=result["confidence"],
            success_threshold=success_threshold,
        )

        ord_hint = local_ord_hint(train_df, current_protocol["aryl_halide"])

        diagnosis = make_diagnosis(
            failure_type=failure_type,
            top_parameter=top_parameter,
            ord_hint=ord_hint,
            protocol=current_protocol,
            yield_percent=result["yield_percent"],
        )

        if result.get("reaction_SMILES"):
            admet = "reaction_SMILES available; full ADMET can be added with RDKit/admet-ai."
        else:
            admet = "not_checked"

        if result["yield_percent"] >= success_threshold:
            cloud_status = "Target reached"
        else:
            cloud_status = "Result received"

        row = {
            "attempt": attempt,
            "cloud_status": cloud_status,
            "protocol": current_protocol.copy(),
            "yield_percent": result["yield_percent"],
            "true_yield_percent": result["true_yield_percent"],
            "failure_type": failure_type,
            "diagnosis": diagnosis,
            "recovery_reason": "Starting low-yield reaction selected from dataset.",
            "admet": admet,
            "ord_hint": ord_hint,
            "protocol_json": protocol_json,
        }

        history.append(row)

        if result["yield_percent"] >= success_threshold:
            history[-1][
                "recovery_reason"
            ] = "Success threshold reached. Campaign complete."
            break

        if attempt == max_attempts:
            history[-1]["recovery_reason"] = "Max attempts reached. Campaign stopped."
            break

        tried_protocols = [h["protocol"] for h in history]

        next_protocol, reason = recover_next_protocol(
            train_df=train_df,
            analyzer=analyzer,
            current_protocol=current_protocol,
            tried_protocols=tried_protocols,
        )

        history[-1]["recovery_reason"] = reason

        current_protocol = next_protocol
        previous_yield = result["yield_percent"]

    history = add_progress_metrics(history)

    return history, importance


# ============================================================
# ITERATION / PROGRESS METRICS
# ============================================================
def add_progress_metrics(history):
    previous_yield = None

    for h in history:
        current_yield = float(h["yield_percent"])

        if previous_yield is None:
            h["yield_point_change"] = 0.0
            h["percent_increase_from_previous"] = 0.0
            h["change_label"] = "Starting point"
        else:
            point_change = current_yield - previous_yield

            if previous_yield == 0:
                percent_change = 0.0
            else:
                percent_change = (point_change / previous_yield) * 100

            h["yield_point_change"] = round(point_change, 2)
            h["percent_increase_from_previous"] = round(percent_change, 2)

            if point_change > 0:
                h["change_label"] = f"+{round(point_change, 2)} yield points"
            elif point_change < 0:
                h["change_label"] = f"{round(point_change, 2)} yield points"
            else:
                h["change_label"] = "No change"

        previous_yield = current_yield

    return history


def build_iteration_progress_table(history):
    rows = []

    for h in history:
        rows.append(
            {
                "Attempt": h["attempt"],
                "Yield %": h["yield_percent"],
                "Yield Point Change": h.get("yield_point_change", 0.0),
                "% Increase From Previous": h.get(
                    "percent_increase_from_previous", 0.0
                ),
                "Failure Type": h["failure_type"],
                "Agent Decision": h["recovery_reason"],
            }
        )

    return pd.DataFrame(rows)


def build_lab_notebook_csv(history):
    rows = []

    for h in history:
        row = {
            "attempt": h["attempt"],
            "yield_percent": h["yield_percent"],
            "true_yield_percent": h["true_yield_percent"],
            "yield_point_change": h.get("yield_point_change", 0.0),
            "percent_increase_from_previous": h.get(
                "percent_increase_from_previous", 0.0
            ),
            "failure_type": h["failure_type"],
            "diagnosis": h["diagnosis"],
            "recovery_reason": h["recovery_reason"],
            "admet": h["admet"],
            "ord_hint": h["ord_hint"],
        }

        for param in REAL_PARAMETERS:
            row[param] = h["protocol"][param]

        rows.append(row)

    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def build_work_order_json(history):
    latest = history[-1]
    payload = {
        "exported_at": datetime.now().isoformat(),
        "latest_attempt": latest["attempt"],
        "latest_protocol_json": latest.get("protocol_json", {}),
        "campaign_history": history,
    }

    return json.dumps(payload, indent=2, default=str).encode("utf-8")


# ============================================================
# VISUALIZATION HELPERS
# ============================================================
def protocol_diff_table(history):
    rows = []

    for param in REAL_PARAMETERS:
        row = {"Parameter": param}
        previous = None

        for h in history:
            col = f"Attempt {h['attempt']}"
            value = str(h["protocol"][param])

            if previous is None:
                row[col] = value
            elif value != previous:
                row[col] = "CHANGED → " + value
            else:
                row[col] = "same → " + value

            previous = value

        rows.append(row)

    reason_row = {"Parameter": "Agent reason"}

    for h in history:
        col = f"Attempt {h['attempt']}"
        reason_row[col] = h["recovery_reason"]

    rows.append(reason_row)

    return pd.DataFrame(rows)


def show_protocol_diff(history):
    df = protocol_diff_table(history)

    def color_cells(value):
        value = str(value)

        if "CHANGED" in value:
            return "background-color: #dcfce7; color: #166534; font-weight: bold;"
        if "same" in value:
            return "background-color: #f3f4f6; color: #374151;"
        if "Starting" in value:
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
        if "Success" in value or "complete" in value:
            return "background-color: #dbeafe; color: #1e40af; font-weight: bold;"
        return ""

    st.dataframe(df.style.map(color_cells), use_container_width=True, hide_index=True)


def show_cloud_lab_steps():
    cols = st.columns(6)

    steps = [
        ("1", "Scientist Goal", "Choose substrate and target yield."),
        ("2", "Protocol Queue", "Generate lab-ready protocol JSON."),
        ("3", "Virtual Execution", "Dataset oracle simulates lab run."),
        ("4", "Yield Result", "Measured yield returns to agent."),
        ("5", "Diagnosis", "Failure type and explanation generated."),
        ("6", "Recovery", "Next condition is proposed."),
    ]

    for col, step in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-number">{step[0]}</div>
                    <div class="workflow-title">{step[1]}</div>
                    <div class="workflow-text">{step[2]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_iteration_yield_graph(history, success_threshold):
    progress_df = build_iteration_progress_table(history)

    chart_df = progress_df.set_index("Attempt")[["Yield %", "% Increase From Previous"]]

    st.line_chart(chart_df)

    st.caption(
        "This graph shows every autonomous attempt, the measured yield, "
        "and the percent increase compared with the previous attempt."
    )

    st.dataframe(progress_df, use_container_width=True, hide_index=True)


def show_attempt_workflow(history):
    st.subheader("Attempt-by-Attempt Workflow")

    for h in history:
        if h["failure_type"] == "success":
            pill = '<span class="pill-green">success</span>'
        elif h["failure_type"] in ["moderate_yield", "near_success"]:
            pill = f'<span class="pill-amber">{h["failure_type"]}</span>'
        elif h["failure_type"] == "low_yield":
            pill = '<span class="pill-red">low yield</span>'
        else:
            pill = f'<span class="pill-blue">{h["failure_type"]}</span>'

        st.markdown('<div class="attempt-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="attempt-title">Attempt {h["attempt"]}</div>
            {pill}
            <div class="small-muted">Cloud status: {h["cloud_status"]}</div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

        with c1:
            st.metric("Yield", f'{h["yield_percent"]}%')

        with c2:
            st.metric("Change", h.get("change_label", "Starting point"))

        with c3:
            st.metric("% Increase", f'{h.get("percent_increase_from_previous", 0.0)}%')

        with c4:
            st.write("**Agent decision**")
            st.write(h["recovery_reason"])

        with st.expander(f"Protocol used in Attempt {h['attempt']}"):
            st.json(h["protocol"])

        st.markdown("</div>", unsafe_allow_html=True)


def show_story_panel(history):
    columns = st.columns(4)

    cards = []

    cards.append(
        {
            "title": "Attempt 1",
            "yield": history[0]["yield_percent"],
            "label": history[0]["failure_type"],
            "note": history[0]["diagnosis"],
        }
    )

    cards.append(
        {
            "title": "Fix Applied",
            "yield": None,
            "label": "recovery",
            "note": history[0]["recovery_reason"],
        }
    )

    if len(history) >= 2:
        cards.append(
            {
                "title": "Attempt 2",
                "yield": history[1]["yield_percent"],
                "label": history[1]["failure_type"],
                "note": history[1]["diagnosis"],
            }
        )

    final = history[-1]

    cards.append(
        {
            "title": "Final",
            "yield": final["yield_percent"],
            "label": final["failure_type"],
            "note": final["diagnosis"],
        }
    )

    for i, card in enumerate(cards[:4]):
        with columns[i]:
            if card["label"] == "success":
                pill = '<span class="pill-green">success</span>'
            elif card["label"] == "recovery":
                pill = '<span class="pill-blue">recovery</span>'
            elif card["label"] in ["moderate_yield", "near_success"]:
                pill = f'<span class="pill-amber">{card["label"]}</span>'
            else:
                pill = f'<span class="pill-red">{card["label"]}</span>'

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{card["title"]}</div>
                    <div class="metric-value">{"N/A" if card["yield"] is None else str(card["yield"]) + "%"}</div>
                    {pill}
                    <div class="metric-note">{card["note"][:115]}{"..." if len(card["note"]) > 115 else ""}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# MAIN UI
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div>
            <span class="tag">Autonomous Labs</span>
            <span class="tag">Cloud Lab Workflow</span>
            <span class="tag">Failed Experiment Recovery</span>
        </div>
        <div class="hero-title">🧪 Cloud Lab Recovery Agent</div>
        <div class="hero-subtitle">
            From failed reaction to next experiment in one click.
            This scientist-facing console automates reaction troubleshooting, yield tracking,
            protocol generation, and next-condition recovery for Buchwald-Hartwig synthesis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
        <b>Problem solved:</b> Scientists waste time after failed reactions manually logging results,
        comparing old experiments, diagnosing failures, and writing the next protocol.
        This tool closes that loop by turning a low-yield result into the next cloud-lab work order.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
        <b>Correct dataset scope:</b> This demo only uses <code>ligand</code>, <code>additive</code>,
        <code>base</code>, and <code>aryl_halide</code>. It does not use temperature, solvent, time,
        or catalyst loading because those fields are not in the uploaded CSV.
    </div>
    """,
    unsafe_allow_html=True,
)


df = None
train_df = None

try:
    df = load_data(CSV_PATH)

    if df is not None:
        train_df = get_train_df(df)

except Exception as error:
    st.error(f"Dataset loading issue: {error}")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Scientist Console")

    show_openai_status()

    use_demo_mode = st.toggle(
        "Demo mode fallback",
        value=False,
        help="Use pre-computed results if live data or OpenAI fails.",
    )

    success_threshold = st.slider(
        "Target yield %",
        min_value=50,
        max_value=95,
        value=int(SUCCESS_THRESHOLD),
        step=5,
    )

    max_attempts = st.slider(
        "Max autonomous attempts",
        min_value=2,
        max_value=6,
        value=MAX_ATTEMPTS,
        step=1,
    )

    if train_df is not None:
        aryl_options = sorted(train_df["aryl_halide"].dropna().unique().tolist())
        selected_aryl = st.selectbox("Target substrate / aryl_halide", aryl_options)

        st.divider()
        st.caption("Dataset Health")
        st.write("Rows:", len(df))
        st.write("Train rows:", len(train_df))
        st.write("Mean yield:", round(float(df["yield"].mean()), 2))
        st.write(
            "Low yield <30%:", f"{round(float((df['yield'] < 30).mean() * 100), 2)}%"
        )
        st.write(
            "High yield ≥80%:", f"{round(float((df['yield'] >= 80).mean() * 100), 2)}%"
        )
    else:
        selected_aryl = "Selected substrate"
        st.warning("CSV not found. Turn on Demo mode fallback.")

    run_button = st.button("Launch Cloud Lab Campaign", type="primary")


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = None

if "importance" not in st.session_state:
    st.session_state.importance = None

if run_button:
    with st.spinner(
        "Queuing protocol, simulating cloud lab execution, and reading yield..."
    ):
        time.sleep(0.8)

        if use_demo_mode or train_df is None:
            st.session_state.history = add_progress_metrics(FALLBACK_HISTORY.copy())
            st.session_state.importance = FALLBACK_IMPORTANCE
        else:
            history, importance = run_live_campaign(
                train_df=train_df,
                selected_aryl_halide=selected_aryl,
                success_threshold=float(success_threshold),
                max_attempts=int(max_attempts),
            )
            st.session_state.history = history
            st.session_state.importance = importance


if st.session_state.history is None:
    st.warning(
        "Choose a target substrate in the sidebar, then click **Launch Cloud Lab Campaign**."
    )
    show_cloud_lab_steps()
    st.stop()


history = st.session_state.history
importance = st.session_state.importance

if history is None or len(history) == 0:
    st.error("No campaign history found. Please launch the campaign again.")
    st.stop()

if importance is None:
    importance = {}


# ============================================================
# TOP METRICS
# ============================================================
start_yield = history[0]["yield_percent"]
final_yield = history[-1]["yield_percent"]
total_gain = final_yield - start_yield

if start_yield == 0:
    total_percent_gain = 0.0
else:
    total_percent_gain = (total_gain / start_yield) * 100

success = final_yield >= float(success_threshold)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Campaign Status</div>
            <div class="metric-value">{"Target Reached" if success else "Needs Review"}</div>
            {'<span class="pill-green">success</span>' if success else '<span class="pill-amber">incomplete</span>'}
            <div class="metric-note">Autonomous loop finished</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Attempts Run</div>
            <div class="metric-value">{len(history)}</div>
            <span class="pill-blue">iterations</span>
            <div class="metric-note">Every attempt is shown below</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Yield Recovery</div>
            <div class="metric-value">{start_yield}% → {final_yield}%</div>
            <span class="pill-green">+{round(total_gain, 2)} points</span>
            <div class="metric-note">{round(total_percent_gain, 2)}% total increase</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Execution Layer</div>
            <div class="metric-value">Virtual Oracle</div>
            <span class="pill-blue">cloud-lab ready</span>
            <div class="metric-note">Replace with robot + yield result</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CLOUD LAB LOOP
# ============================================================
st.subheader("Autonomous Cloud Lab Workflow")
show_cloud_lab_steps()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Run Monitor",
        "Protocol Delta",
        "Scientist Notebook",
        "Cloud Lab Handoff",
        "Exports",
    ]
)


with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Iteration Yield Progress")
        show_iteration_yield_graph(history, float(success_threshold))

    with right:
        st.subheader("Parameter Importance")

        if importance:
            importance_df = pd.DataFrame(
                {
                    "Parameter": list(importance.keys()),
                    "Importance Score": list(importance.values()),
                }
            ).set_index("Parameter")

            st.bar_chart(importance_df)
        else:
            st.warning("No importance values available.")

    st.divider()
    show_attempt_workflow(history)

    st.divider()
    st.subheader("Diagnosis Story")
    show_story_panel(history)


with tab2:
    st.subheader("Protocol Diff Table")
    st.caption(
        "This is the main judge artifact: what changed, when it changed, and why the agent changed it."
    )
    show_protocol_diff(history)


with tab3:
    st.subheader("Scientist Notebook")

    st.markdown(
        """
        This tab acts like an automatic lab notebook.
        It records the protocol, yield result, diagnosis, and next recovery decision for every attempt.
        """
    )

    for h in history:
        with st.expander(
            f"Attempt {h['attempt']} — {h['failure_type']} — yield {h['yield_percent']}%",
            expanded=(h["attempt"] == history[-1]["attempt"]),
        ):
            st.markdown("**Cloud lab status**")
            st.write(h["cloud_status"])

            st.markdown("**Protocol**")
            st.json(h["protocol"])

            st.markdown("**Measurement and progress**")
            st.write(
                {
                    "measured_yield_percent": h["yield_percent"],
                    "oracle_true_yield_percent": h["true_yield_percent"],
                    "yield_point_change": h.get("yield_point_change", 0.0),
                    "percent_increase_from_previous": h.get(
                        "percent_increase_from_previous", 0.0
                    ),
                    "admet_status": h["admet"],
                    "ord_style_hint": h["ord_hint"],
                }
            )

            st.markdown("**Diagnosis**")
            st.write(h["diagnosis"])

            st.markdown("**Recovery decision**")
            st.write(h["recovery_reason"])


with tab4:
    st.subheader("Cloud Lab Handoff")

    st.markdown(
        """
        This section shows what would be sent to a cloud lab execution layer.
        In this hackathon demo, the execution layer is the dataset oracle.
        In a real lab, this JSON could become a robot queue item or technician work order.
        """
    )

    latest = history[-1]

    st.markdown("**Latest protocol JSON**")
    st.json(latest.get("protocol_json", {}))

    st.markdown("**Real deployment mapping**")

    deployment_df = pd.DataFrame(
        [
            {
                "Demo Component": "Protocol JSON",
                "Real Cloud Lab Equivalent": "Robot or technician work order",
            },
            {
                "Demo Component": "CSV oracle",
                "Real Cloud Lab Equivalent": "Physical experiment + yield measurement",
            },
            {
                "Demo Component": "Failure classifier",
                "Real Cloud Lab Equivalent": "Automated result interpretation",
            },
            {
                "Demo Component": "Recovery agent",
                "Real Cloud Lab Equivalent": "Next experiment queued without manual redesign",
            },
            {
                "Demo Component": "Scientist notebook",
                "Real Cloud Lab Equivalent": "Automatic ELN / Airtable / Benchling logging",
            },
        ]
    )

    st.dataframe(deployment_df, use_container_width=True, hide_index=True)


with tab5:
    st.subheader("Exports")

    st.markdown(
        """
        These exports make the demo feel like a real workflow automation tool.
        The CSV can act as a lab notebook or Airtable import.
        The JSON can act as a cloud-lab work order.
        """
    )

    csv_bytes = build_lab_notebook_csv(history)
    json_bytes = build_work_order_json(history)

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download lab notebook CSV",
            data=csv_bytes,
            file_name="cloud_lab_recovery_log.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            label="Download cloud lab work order JSON",
            data=json_bytes,
            file_name="cloud_lab_work_order.json",
            mime="application/json",
        )

    st.markdown("**Airtable-ready preview**")
    st.dataframe(
        build_iteration_progress_table(history),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PHASE 5 CHECKLIST
# ============================================================
with st.expander("Phase 5 Reliability Checklist", expanded=True):
    st.checkbox("Live campaign can run from CSV", value=train_df is not None)
    st.checkbox("Demo fallback mode available", value=True)
    st.checkbox("OpenAI used only for diagnosis explanation", value=True)
    st.checkbox("No fake parameters used", value=True)
    st.checkbox("Iteration graph shows every attempt and yield increase", value=True)
    st.checkbox("Cloud lab handoff shown as protocol JSON", value=True)
    st.checkbox(
        "CSV/JSON exports available for Airtable or lab notebook workflow", value=True
    )


st.success(
    "Demo ready: scientist selects substrate → cloud lab simulates run → agent diagnoses → recovery protocol is queued."
)
