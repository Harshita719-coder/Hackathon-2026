import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CSV_PATH = r"C:\Users\vrush\Downloads\buchwald_hartwig_huggingface.csv"

REAL_PARAMETERS = ["ligand", "additive", "base", "aryl_halide"]
SUCCESS_THRESHOLD = 80.0
MAX_ATTEMPTS = 3
NOISE_STD_PERCENT = 3.0


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Autonomous Synthesis Recovery Agent",
    page_icon="🧪",
    layout="wide",
)


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        st.error(f"CSV file not found: {csv_path}")
        st.stop()

    df = pd.read_csv(path)

    required_columns = [
        "ligand",
        "additive",
        "base",
        "aryl_halide",
        "yield",
        "split",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    return df


df = load_data(CSV_PATH)
train_df = df[df["split"] == "train"].copy()

if train_df.empty:
    st.error("Train split is empty. Please check your CSV file.")
    st.stop()


# ----------------------------
# Helper logic
# ----------------------------
class BHOracle:
    """
    Virtual lab oracle.

    It simulates a lab experiment using only real dataset fields:
    ligand, additive, base, aryl_halide.
    """

    def __init__(self, train_df: pd.DataFrame, noise_std_percent: float = 3.0, seed: int = 42):
        self.train_df = train_df.copy()
        self.noise_std_percent = noise_std_percent
        self.rng = np.random.default_rng(seed)

    def query(self, protocol: dict) -> dict:
        exact = self.train_df[
            (self.train_df["ligand"] == protocol["ligand"])
            & (self.train_df["additive"] == protocol["additive"])
            & (self.train_df["base"] == protocol["base"])
            & (self.train_df["aryl_halide"] == protocol["aryl_halide"])
        ]

        if len(exact) > 0:
            true_yield = float(exact["yield"].mean())
            confidence = 0.95
            source = "exact_train_match"
            matched_rows = len(exact)
            reaction_smiles = exact["reaction_SMILES"].iloc[0] if "reaction_SMILES" in exact.columns else None
        else:
            fallback = self.train_df[
                (self.train_df["base"] == protocol["base"])
                & (self.train_df["aryl_halide"] == protocol["aryl_halide"])
            ]

            if len(fallback) > 0:
                true_yield = float(fallback["yield"].mean())
                confidence = 0.65
                source = "fallback_same_base_and_aryl_halide"
                matched_rows = len(fallback)
                reaction_smiles = fallback["reaction_SMILES"].iloc[0] if "reaction_SMILES" in fallback.columns else None
            else:
                true_yield = float(self.train_df["yield"].mean())
                confidence = 0.35
                source = "fallback_train_average"
                matched_rows = len(self.train_df)
                reaction_smiles = None

        noisy_yield = true_yield + self.rng.normal(0, self.noise_std_percent)
        noisy_yield = float(np.clip(noisy_yield, 0, 100))

        return {
            **protocol,
            "yield_percent": round(noisy_yield, 2),
            "true_yield_percent": round(true_yield, 2),
            "confidence": confidence,
            "source": source,
            "matched_rows": matched_rows,
            "reaction_SMILES": reaction_smiles,
        }


class SimpleAnalyzer:
    """
    Lightweight importance analyzer.

    This avoids sklearn/scipy. It estimates importance by measuring
    how much average yield changes across each parameter.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()
        self.features = REAL_PARAMETERS

    def feature_importance(self) -> dict:
        importance = {}

        for feature in self.features:
            grouped = self.train_df.groupby(feature)["yield"].mean()

            if len(grouped) <= 1:
                score = 0.0
            else:
                score = float(grouped.max() - grouped.min())

            importance[feature] = round(score, 2)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def predict_yield(self, protocol: dict) -> float:
        """
        Simple predicted yield.

        Uses marginal averages for each condition choice.
        It does not directly sort by the exact target row yield.
        """

        estimates = []

        for feature in self.features:
            subset = self.train_df[self.train_df[feature] == protocol[feature]]

            if len(subset) > 0:
                estimates.append(float(subset["yield"].mean()))

        if estimates:
            return round(float(np.mean(estimates)), 2)

        return round(float(self.train_df["yield"].mean()), 2)


def classify_failure(yield_percent: float, previous_yield=None, confidence=0.95) -> str:
    if confidence < 0.5:
        return "high_uncertainty"

    if previous_yield is not None:
        improvement = yield_percent - previous_yield
        if improvement < 5 and yield_percent < SUCCESS_THRESHOLD:
            return "no_improvement"

    if yield_percent < 30:
        return "low_yield"

    if 30 <= yield_percent < 60:
        return "moderate_yield"

    if 60 <= yield_percent < 80:
        return "near_success"

    return "success"


def admet_placeholder(reaction_smiles):
    """
    Safe demo ADMET placeholder.

    We do not require RDKit here so the Streamlit demo does not break.
    """

    if reaction_smiles:
        return {
            "admet_pass": "not_checked",
            "flags": "reaction_SMILES found; full ADMET can be added with RDKit/admet-ai.",
        }

    return {
        "admet_pass": "not_checked",
        "flags": "No reaction_SMILES available.",
    }


def local_ord_check(train_df: pd.DataFrame, aryl_halide: str) -> dict:
    """
    Local ORD-style check.

    Finds the most common base in high-yield reactions for the same aryl_halide.
    """

    subset = train_df[
        (train_df["aryl_halide"] == aryl_halide)
        & (train_df["yield"] >= 70)
    ]

    if subset.empty:
        return {
            "most_common_successful_base": "None found",
            "support_count": 0,
        }

    counts = subset["base"].value_counts()

    return {
        "most_common_successful_base": str(counts.index[0]),
        "support_count": int(counts.iloc[0]),
    }


def make_diagnosis(failure_type: str, top_parameter: str, ord_result: dict) -> str:
    base_hint = ord_result.get("most_common_successful_base", "None found")

    messages = {
        "low_yield": (
            "Low yield detected. The agent should recover by changing the available "
            "reaction choices: ligand, additive, and base."
        ),
        "moderate_yield": (
            "The reaction is partially working, but the yield is not high enough. "
            "The agent should try a stronger condition set."
        ),
        "near_success": (
            "The reaction is close to the target. The agent should make a smaller recovery step."
        ),
        "no_improvement": (
            "The new attempt did not improve enough. The agent should explore a different condition."
        ),
        "high_uncertainty": (
            "The result has low confidence. The agent should use a better-supported condition."
        ),
        "success": (
            "The target yield was reached. The autonomous loop can stop."
        ),
    }

    diagnosis = messages.get(failure_type, "The agent selected the next valid protocol.")

    diagnosis += f" Current top data lever: {top_parameter}."

    if base_hint != "None found":
        diagnosis += f" Local ORD-style check supports base choice: {base_hint}."

    return diagnosis


def find_starting_protocol(train_df: pd.DataFrame, selected_aryl_halide: str) -> dict:
    subset = train_df[train_df["aryl_halide"] == selected_aryl_halide].copy()

    low_subset = subset[subset["yield"] < 30].copy()

    if not low_subset.empty:
        row = low_subset.sample(1, random_state=42).iloc[0]
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


def recover_next_protocol(
    train_df: pd.DataFrame,
    analyzer: SimpleAnalyzer,
    current_protocol: dict,
    tried_protocols: list,
) -> dict:
    same_aryl = train_df[train_df["aryl_halide"] == current_protocol["aryl_halide"]].copy()

    if same_aryl.empty:
        same_aryl = train_df.copy()

    tried_set = set()

    for p in tried_protocols:
        tried_set.add((p["ligand"], p["additive"], p["base"], p["aryl_halide"]))

    candidates = []

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

        if key in tried_set:
            continue

        predicted_yield = analyzer.predict_yield(candidate)

        candidates.append(
            {
                "protocol": candidate,
                "predicted_yield": predicted_yield,
            }
        )

    if not candidates:
        return {
            "next_protocol": current_protocol,
            "predicted_yield": analyzer.predict_yield(current_protocol),
            "reason": "No untried candidate found; retrying current condition.",
        }

    candidates = sorted(candidates, key=lambda x: x["predicted_yield"], reverse=True)
    best = candidates[0]

    changes = []

    for param in REAL_PARAMETERS:
        if current_protocol[param] != best["protocol"][param]:
            changes.append(param)

    if changes:
        change_text = ", ".join(changes)
    else:
        change_text = "no parameter"

    return {
        "next_protocol": best["protocol"],
        "predicted_yield": best["predicted_yield"],
        "reason": f"Recovery agent changed {change_text} using simple model-predicted yield.",
    }


def run_campaign(selected_aryl_halide: str):
    oracle = BHOracle(train_df, noise_std_percent=NOISE_STD_PERCENT)
    analyzer = SimpleAnalyzer(train_df)

    importance = analyzer.feature_importance()
    top_parameter = list(importance.keys())[0] if importance else "unknown"

    current_protocol = find_starting_protocol(train_df, selected_aryl_halide)

    history = []
    previous_yield = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = oracle.query(current_protocol)
        failure_type = classify_failure(
            result["yield_percent"],
            previous_yield=previous_yield,
            confidence=result["confidence"],
        )

        ord_result = local_ord_check(train_df, current_protocol["aryl_halide"])
        admet_result = admet_placeholder(result.get("reaction_SMILES"))
        diagnosis = make_diagnosis(failure_type, top_parameter, ord_result)

        row = {
            "attempt": attempt,
            "protocol": current_protocol.copy(),
            "result": result,
            "yield_percent": result["yield_percent"],
            "true_yield_percent": result["true_yield_percent"],
            "failure_type": failure_type,
            "diagnosis": diagnosis,
            "admet_pass": admet_result["admet_pass"],
            "admet_flags": admet_result["flags"],
            "ord_base": ord_result["most_common_successful_base"],
            "ord_support_count": ord_result["support_count"],
            "recovery_reason": "Starting failed reaction selected from dataset." if attempt == 1 else "",
            "predicted_next_yield": None,
        }

        history.append(row)

        if result["yield_percent"] >= SUCCESS_THRESHOLD:
            break

        if attempt == MAX_ATTEMPTS:
            break

        tried_protocols = [h["protocol"] for h in history]

        recovery = recover_next_protocol(
            train_df=train_df,
            analyzer=analyzer,
            current_protocol=current_protocol,
            tried_protocols=tried_protocols,
        )

        history[-1]["recovery_reason"] = recovery["reason"]
        history[-1]["predicted_next_yield"] = recovery["predicted_yield"]

        current_protocol = recovery["next_protocol"]
        previous_yield = result["yield_percent"]

    return {
        "history": history,
        "importance": importance,
    }


# ----------------------------
# Visualization helpers
# ----------------------------
def build_yield_chart(history):
    attempts = [h["attempt"] for h in history]
    yields = [h["yield_percent"] for h in history]

    upper = [min(100, y + NOISE_STD_PERCENT) for y in yields]
    lower = [max(0, y - NOISE_STD_PERCENT) for y in yields]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=attempts + attempts[::-1],
            y=upper + lower[::-1],
            fill="toself",
            name="Simulated ± noise band",
            line=dict(width=0),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=attempts,
            y=yields,
            mode="lines+markers",
            name="Observed yield %",
        )
    )

    fig.add_hline(
        y=SUCCESS_THRESHOLD,
        line_dash="dash",
        annotation_text="80% success target",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Yield Convergence",
        xaxis_title="Attempt",
        yaxis_title="Yield %",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def build_importance_chart(importance):
    items = list(importance.items())[:5]
    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
        )
    )

    top_label = labels[0] if labels else "unknown"

    fig.update_layout(
        title=f"Parameter Importance: {top_label} is the #1 lever in this dataset",
        xaxis_title="Yield spread score",
        yaxis_title="Parameter",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_yaxes(autorange="reversed")

    return fig


def build_protocol_diff_df(history):
    rows = []

    for param in REAL_PARAMETERS:
        row = {"Parameter": param}

        previous_value = None

        for h in history:
            attempt_label = f"Attempt {h['attempt']}"
            value = h["protocol"][param]

            if previous_value is None:
                display_value = str(value)
            elif value != previous_value:
                display_value = f"↑ changed → {value}"
            else:
                display_value = f"same → {value}"

            row[attempt_label] = display_value
            previous_value = value

        rows.append(row)

    why_row = {"Parameter": "Why / Agent reason"}

    for h in history:
        attempt_label = f"Attempt {h['attempt']}"
        why_row[attempt_label] = h["recovery_reason"] if h["recovery_reason"] else h["diagnosis"]

    rows.append(why_row)

    return pd.DataFrame(rows)


def style_protocol_diff(df_diff):
    def style_cell(value):
        value = str(value)

        if "changed" in value:
            return "background-color: #d1fae5; color: #065f46; font-weight: 600;"
        if "same" in value:
            return "background-color: #f3f4f6; color: #374151;"
        if "Starting failed" in value:
            return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
        if "Why" in value:
            return "font-weight: 700;"
        return ""

    return df_diff.style.map(style_cell)


def show_story_cards(history):
    cols = st.columns(4)

    card_data = []

    if len(history) >= 1:
        card_data.append(
            {
                "title": "Attempt 1",
                "status": "Failed / Starting point",
                "yield": history[0]["yield_percent"],
                "failure": history[0]["failure_type"],
                "note": history[0]["diagnosis"],
            }
        )

    if len(history) >= 1:
        card_data.append(
            {
                "title": "Fix Applied",
                "status": "Recovery decision",
                "yield": history[0].get("predicted_next_yield"),
                "failure": "agent_fix",
                "note": history[0]["recovery_reason"],
            }
        )

    if len(history) >= 2:
        card_data.append(
            {
                "title": "Attempt 2",
                "status": "Retest",
                "yield": history[1]["yield_percent"],
                "failure": history[1]["failure_type"],
                "note": history[1]["diagnosis"],
            }
        )

    if len(history) >= 3:
        final = history[2]
    else:
        final = history[-1]

    card_data.append(
        {
            "title": "Final Attempt",
            "status": "Success" if final["yield_percent"] >= SUCCESS_THRESHOLD else "Stopped",
            "yield": final["yield_percent"],
            "failure": final["failure_type"],
            "note": final["diagnosis"],
        }
    )

    for i, card in enumerate(card_data[:4]):
        with cols[i]:
            st.markdown(f"### {card['title']}")
            st.metric("Yield / score", "N/A" if card["yield"] is None else f"{card['yield']}%")
            st.write(f"**Status:** {card['status']}")
            st.write(f"**Label:** {card['failure']}")
            st.caption(card["note"])


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🧪 Demo Controls")

st.sidebar.markdown(
    """
This demo uses the real Buchwald-Hartwig dataset fields:

- ligand
- additive
- base
- aryl_halide
"""
)

aryl_options = sorted(train_df["aryl_halide"].dropna().unique().tolist())

selected_aryl = st.sidebar.selectbox(
    "Choose aryl_halide / substrate",
    aryl_options,
)

run_button = st.sidebar.button("Run optimization", type="primary")

st.sidebar.divider()

st.sidebar.write("Dataset rows:", len(df))
st.sidebar.write("Train rows:", len(train_df))
st.sidebar.write("Mean yield:", round(float(df["yield"].mean()), 2))
st.sidebar.write("Low yield <30%:", f"{round(float((df['yield'] < 30).mean() * 100), 2)}%")
st.sidebar.write("High yield ≥80%:", f"{round(float((df['yield'] >= 80).mean() * 100), 2)}%")


# ----------------------------
# Main page
# ----------------------------
st.title("Autonomous Synthesis Recovery Agent")

st.markdown(
    """
This Streamlit demo shows a judge the autonomous lab loop:

**failed reaction → oracle test → failure diagnosis → recovery decision → next experiment**

The real lab issue is that researchers spend time testing many reaction-condition combinations.
This app shows how an agent can reduce that friction by using previous high-throughput reaction data
to propose the next experiment.
"""
)

st.info(
    "Correct dataset scope: this demo optimizes ligand, additive, base, and aryl_halide. "
    "It does not use temperature, time, solvent, or catalyst loading because those fields are not in the uploaded CSV."
)

if "campaign" not in st.session_state:
    st.session_state.campaign = None

if run_button:
    with st.spinner("Running autonomous recovery loop..."):
        # Tiny pause so judges can see something happening.
        time.sleep(0.5)
        st.session_state.campaign = run_campaign(selected_aryl)

if st.session_state.campaign is None:
    st.warning("Select an aryl_halide in the sidebar, then click **Run optimization**.")
    st.stop()

campaign = st.session_state.campaign
history = campaign["history"]
importance = campaign["importance"]

# Top metrics
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Attempts", len(history))

with m2:
    st.metric("Starting yield", f"{history[0]['yield_percent']}%")

with m3:
    st.metric("Final yield", f"{history[-1]['yield_percent']}%")

with m4:
    success_text = "Yes" if history[-1]["yield_percent"] >= SUCCESS_THRESHOLD else "No"
    st.metric("Reached 80% target?", success_text)

# 2x2 grid top
left, right = st.columns(2)

with left:
    st.plotly_chart(build_yield_chart(history), use_container_width=True)

with right:
    st.plotly_chart(build_importance_chart(importance), use_container_width=True)

# Protocol diff full width
st.subheader("Protocol Diff Table")

st.caption(
    "This is the main demo artifact. It shows what changed across attempts and why the agent made the change."
)

diff_df = build_protocol_diff_df(history)
st.dataframe(style_protocol_diff(diff_df), use_container_width=True, hide_index=True)

# Diagnosis story
with st.expander("Diagnosis Story Panel", expanded=True):
    show_story_cards(history)

# Detailed attempt logs
with st.expander("Raw Attempt Log"):
    for h in history:
        st.markdown(f"### Attempt {h['attempt']}")
        st.json(
            {
                "protocol": h["protocol"],
                "yield_percent": h["yield_percent"],
                "true_yield_percent": h["true_yield_percent"],
                "failure_type": h["failure_type"],
                "diagnosis": h["diagnosis"],
                "admet_pass": h["admet_pass"],
                "admet_flags": h["admet_flags"],
                "ord_style_base_hint": h["ord_base"],
                "recovery_reason": h["recovery_reason"],
            }
        )

st.success("Phase 4 demo complete. This is ready to show as the live judge-facing surface.")