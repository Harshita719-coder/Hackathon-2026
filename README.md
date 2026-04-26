# CloudLabs — Autonomous Synthesis Recovery Agent

An AI-powered lab optimization loop that autonomously diagnoses and recovers failed chemical synthesis experiments. Built for the SCSP Hackathon 2026.

**Live demo:** https://hackathon-2026-8h29pqvfwqrqpa9nxmfpel.streamlit.app/

---

## Team

**Team Name:** Crystal Hackers

| Name |
|------|
| Sehaj Gill |
| Harshita LNU|
| Vrusha Patel |
| Shubham Kalia |

## Track

Autonomous Laboratories

---

## What it does

Given a low-yield Buchwald-Hartwig cross-coupling reaction, the agent:

1. Selects starting conditions from a real experimental dataset
2. Queries a virtual lab oracle to evaluate the reaction
3. Diagnoses the failure mode (wrong ligand, base mismatch, etc.)
4. Proposes improved reaction parameters using an ML analyzer
5. Retries up to 3 times, targeting ≥ 80% yield

## Features

- **Yield Convergence Chart** — Plotly visualization of yield improvement across attempts with confidence band
- **Parameter Importance Chart** — Which reaction parameters (ligand, base, additive, aryl halide) drove the improvement
- **Protocol Diff Table** — Side-by-side comparison of parameter changes and agent reasoning per attempt
- **Diagnosis Story Panel** — Narrative cards summarizing each attempt's outcome
- **Raw Attempt Log** — Expandable JSON logs with full protocol, yields, failure classification, and diagnosis

## Datasets & APIs Used

| Resource | Purpose |
|----------|---------|
| Buchwald-Hartwig dataset (HuggingFace) | Real experimental reaction conditions and yields |
| OpenAI API | Failure diagnosis, parameter reasoning, and recovery suggestions |

## Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit, Plotly |
| AI | OpenAI API |
| Data | Pandas, NumPy |
| Dataset | Buchwald-Hartwig (HuggingFace) |

## Running locally

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Then run:

```bash
streamlit run app_phase5.py
```

