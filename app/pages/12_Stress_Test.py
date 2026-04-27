from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.decision_engine import (
    build_city_readiness_summary,
    build_escalation_plan,
    build_sector_actions,
    readiness_to_color,
)
from src.escalation_engine import predict_escalation_from_features
from src.forecast_engine import make_ml_forecast
from src.impact_engine import (
    build_operational_triggers,
    identify_primary_impacts,
    identify_priority_groups,
    impact_band_from_peak,
)
from src.resource_routing_engine import (
    build_top_dispatch_summary,
    recommend_dispatch_resources,
)
from src.sidebar import render_app_sidebar
from src.vulnerability_engine import (
    build_impact_adjusted_priority,
    build_vulnerability_recommendations,
    get_city_vulnerability_snapshot,
    identify_vulnerability_drivers,
)
from src.xai_engine import explain_escalation_row

st.set_page_config(page_title="Stress Test", page_icon="🔥", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}

READINESS_MAP = {
    "Nizak": "Monitoring",
    "Umjeren": "Prepared",
    "Visok": "Elevated Readiness",
    "Vrlo visok": "Critical Preparedness",
}

CONSENSUS_COLOR_MAP = {
    "Strong stress consensus": "#2E8B57",
    "Moderate stress consensus": "#E6A700",
    "Mixed stress signals": "#E67E22",
    "Low confidence stress signal": "#C0392B",
}

CONFIDENCE_COLOR_MAP = {
    "High": "#2E8B57",
    "Moderate": "#E6A700",
    "Low": "#C0392B",
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .page-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #7c2d12 100%);
        border-radius: 22px;
        padding: 1.4rem 1.55rem 1.2rem 1.55rem;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.16);
    }

    .page-hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .page-hero-subtitle {
        font-size: 0.99rem;
        line-height: 1.62;
        opacity: 0.95;
        max-width: 1080px;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }

    .chip {
        display: inline-block;
        padding: 0.36rem 0.72rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: white;
        font-size: 0.86rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .control-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        min-height: 112px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 0.15rem;
        word-break: break-word;
    }

    .metric-sub {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.5;
    }

    .soft-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        height: 100%;
    }

    .panel-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.7rem;
    }

    .panel-text {
        color: #334155;
        line-height: 1.68;
        font-size: 0.95rem;
    }

    .soft-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.72;
        font-size: 0.95rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.35rem 0 0.85rem 0;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.88rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .note-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #0f172a;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
        line-height: 1.65;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #7c2d12;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
        line-height: 1.65;
    }

    .summary-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        color: #334155;
        line-height: 1.7;
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.72rem 1rem;
        font-weight: 700;
        border: none;
        background: linear-gradient(135deg, #0f172a 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(220, 38, 38, 0.22);
    }

    div.stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(220, 38, 38, 0.28);
        background: linear-gradient(135deg, #1e293b 0%, #ef4444 100%);
        color: white;
    }

    div[data-baseweb="tab-list"] {
        gap: 1.2rem;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
    }

    .stDataFrame, .stPlotlyChart {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_float(value: Any, decimals: int = 1, default: str = "N/A") -> str:
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{decimals}f}"
    except Exception:
        return default


def badge(text: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{text}</span>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list_card(title: str, items: list[str]) -> None:
    list_html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">{title}</div>
            <ul class="soft-list">
                {list_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">{title}</div>
            <div class="panel-text">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def risk_level_from_score(score: float) -> str:
    if score >= 75:
        return "Vrlo visok"
    if score >= 50:
        return "Visok"
    if score >= 25:
        return "Umjeren"
    return "Nizak"


def clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def build_stress_score(
    temp_max: float,
    apparent_temp_max: float,
    humidity_mean: float,
    wind_speed_mean: float,
    persistence_day_index: int,
    power_outage: bool,
) -> float:
    temp_component = max(0.0, (temp_max - 26.0) * 2.5)
    apparent_component = max(0.0, (apparent_temp_max - 28.0) * 2.3)
    humidity_extreme_component = abs(humidity_mean - 50.0) * 0.22
    low_wind_component = max(0.0, (6.0 - wind_speed_mean) * 3.0)
    persistence_component = max(0.0, (persistence_day_index - 1) * 5.5)
    outage_component = 12.0 if power_outage else 0.0

    raw_score = (
        temp_component
        + apparent_component
        + humidity_extreme_component
        + low_wind_component
        + persistence_component
        + outage_component
    )
    return round(clamp_0_100(raw_score), 2)


def apply_stress_overrides(
    forecast_df: pd.DataFrame,
    forced_temp_max: float,
    forced_temp_min: float,
    forced_apparent_extra: float,
    forced_humidity: float,
    forced_wind: float,
    persistence_days: int,
    power_outage: bool,
) -> pd.DataFrame:
    out = forecast_df.copy().sort_values("date").reset_index(drop=True)
    affected_days = min(persistence_days, len(out))

    for i in range(affected_days):
        apparent_temp = forced_temp_max + forced_apparent_extra
        temp_mean = (forced_temp_max + forced_temp_min) / 2.0
        apparent_mean = apparent_temp - 1.2

        score = build_stress_score(
            temp_max=forced_temp_max,
            apparent_temp_max=apparent_temp,
            humidity_mean=forced_humidity,
            wind_speed_mean=forced_wind,
            persistence_day_index=i + 1,
            power_outage=power_outage,
        )
        level = risk_level_from_score(score)

        out.loc[i, "temp_max"] = forced_temp_max
        if "temp_min" in out.columns:
            out.loc[i, "temp_min"] = forced_temp_min
        if "temp_mean" in out.columns:
            out.loc[i, "temp_mean"] = temp_mean
        out.loc[i, "apparent_temp_max"] = apparent_temp
        if "apparent_temp_mean" in out.columns:
            out.loc[i, "apparent_temp_mean"] = apparent_mean
        out.loc[i, "humidity_mean"] = forced_humidity
        if "wind_speed_mean" in out.columns:
            out.loc[i, "wind_speed_mean"] = forced_wind

        out.loc[i, "heuristic_risk_score"] = score
        out.loc[i, "heuristic_risk_level"] = level

        if "heat_risk_score" in out.columns:
            out.loc[i, "heat_risk_score"] = score

        if "ml_predicted_label" in out.columns:
            out.loc[i, "ml_predicted_label"] = level

        if "ml_prediction_confidence" in out.columns:
            out.loc[i, "ml_prediction_confidence"] = min(0.99, 0.62 + abs(score - 50.0) / 100.0)

    return out


def build_stress_consensus(
    stress_first_row: pd.Series,
    escalation_label: str | None,
    escalation_probability: float | None,
    power_outage: bool,
    persistence_days: int,
) -> dict:
    v1_signal = str(stress_first_row.get("heuristic_risk_level", "N/A"))
    v3_signal = escalation_label or "Stable"

    v1_rank = {"Nizak": 0, "Umjeren": 1, "Visok": 2, "Vrlo visok": 3}.get(v1_signal, 0)
    v3_rank = {"Stable": 0, "Watch": 1, "Likely escalation": 2}.get(v3_signal, 0)

    if v1_rank >= 2 and v3_rank >= 1:
        consensus_status = "Strong stress consensus"
    elif v1_rank >= 1 and v3_rank >= 0:
        consensus_status = "Moderate stress consensus"
    else:
        consensus_status = "Mixed stress signals"

    if escalation_probability is None:
        confidence_level = "Low"
    elif escalation_probability >= 0.70 or escalation_probability <= 0.20:
        confidence_level = "High"
    elif escalation_probability >= 0.55 or escalation_probability <= 0.30:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    if power_outage and persistence_days >= 3:
        uncertainty_warning = (
            "Stress scenario includes degraded infrastructure and multi-day persistence. "
            "Operator review is strongly recommended."
        )
        operator_review_required = True
    elif consensus_status == "Mixed stress signals":
        uncertainty_warning = (
            "Stress scenario produces mixed signals between synthetic risk override and escalation model."
        )
        operator_review_required = True
    elif confidence_level == "Low":
        uncertainty_warning = (
            "Escalation probability is in a less certain range for this synthetic scenario."
        )
        operator_review_required = True
    else:
        uncertainty_warning = "Stress scenario signal is operationally usable without major additional warning."
        operator_review_required = False

    return {
        "v1_signal": v1_signal,
        "v3_signal": v3_signal,
        "consensus_status": consensus_status,
        "confidence_level": confidence_level,
        "uncertainty_warning": uncertainty_warning,
        "operator_review_required": operator_review_required,
    }


def build_stress_brief(
    city: str,
    baseline_summary: dict,
    stress_summary: dict,
    stress_consensus: dict,
    vulnerability_snapshot: dict,
    impact_adjusted_priority: float,
    top_dispatch_summary: str,
    xai_summary: dict,
    forced_temp_max: float,
    forced_humidity: float,
    forced_wind: float,
    persistence_days: int,
    power_outage: bool,
) -> str:
    xai_probability = xai_summary.get("probability")
    xai_probability_text = format_float(xai_probability, decimals=2)

    return f"""HEATSAFE HR — STRESS TEST BRIEF

City: {city}

SCENARIO INPUT
- Forced max temperature: {forced_temp_max:.1f} °C
- Forced humidity: {forced_humidity:.1f} %
- Forced wind speed: {forced_wind:.1f} m/s
- Persistence days: {persistence_days}
- Power outage / degraded infrastructure: {"Yes" if power_outage else "No"}

BASELINE
- Baseline readiness: {baseline_summary["readiness_status"]}
- Baseline 7d peak: {baseline_summary["next_7d_peak_level"]} ({baseline_summary["next_7d_peak_score"]:.1f})

STRESS TEST OUTPUT
- Stress readiness: {stress_summary["readiness_status"]}
- Stress 7d peak: {stress_summary["next_7d_peak_level"]} ({stress_summary["next_7d_peak_score"]:.1f})
- Peak date: {stress_summary["next_7d_peak_date"].strftime("%d.%m.%Y.")}

CONSENSUS / RELIABILITY
- Synthetic v1 signal: {stress_consensus["v1_signal"]}
- v3 escalation signal: {stress_consensus["v3_signal"]}
- Consensus status: {stress_consensus["consensus_status"]}
- Confidence level: {stress_consensus["confidence_level"]}
- Uncertainty warning: {stress_consensus["uncertainty_warning"]}
- Operator review required: {"Yes" if stress_consensus["operator_review_required"] else "No"}

VULNERABILITY
- Vulnerability index: {vulnerability_snapshot["vulnerability_index"]:.1f}
- Vulnerability band: {vulnerability_snapshot["vulnerability_band"]}
- Impact-adjusted priority: {impact_adjusted_priority:.1f}

XAI SUMMARY
- Method: {xai_summary.get("method", "N/A")}
- Probability: {xai_probability_text}
- Label: {xai_summary.get("label", "N/A")}
- Summary: {xai_summary.get("explanation_text", "N/A")}

TOP DISPATCH RESOURCE
- {top_dispatch_summary}
"""


st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🔥 Stress Test Simulator</div>
        <div class="page-hero-subtitle">
            What-if simulacija za ekstremne toplinske scenarije. Ova stranica testira kako se sustav ponaša
            kada grad uđe u sintetički “worst-case” režim: ekstremna temperatura, višednevna izloženost,
            slab vjetar i potencijalno degradirana infrastruktura.
        </div>
        <div class="chip-row">
            <span class="chip">Stress Simulation</span>
            <span class="chip">Operational Readiness</span>
            <span class="chip">Reliability Layer</span>
            <span class="chip">Dispatch Support</span>
            <span class="chip">What-If Analysis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = CITIES.index(default_city) if default_city in CITIES else 0

st.markdown('<div class="section-title">Stress scenario control panel</div>', unsafe_allow_html=True)

st.markdown('<div class="control-card">', unsafe_allow_html=True)

c1, c2 = st.columns([1.2, 1])
with c1:
    selected_city = st.selectbox("Odaberi grad", CITIES, index=default_index)
    st.session_state.selected_city = selected_city
with c2:
    power_outage = st.toggle("Power outage / degraded infrastructure", value=False)

s1, s2, s3, s4 = st.columns(4)
with s1:
    forced_temp_max = st.slider("Forced max temp (°C)", 30, 48, 44, 1)
with s2:
    forced_temp_min = st.slider("Forced min temp (°C)", 16, 34, 27, 1)
with s3:
    forced_humidity = st.slider("Forced humidity (%)", 5, 95, 18, 1)
with s4:
    forced_wind = st.slider("Forced wind speed (m/s)", 0, 12, 1, 1)

s5, s6 = st.columns(2)
with s5:
    forced_apparent_extra = st.slider("Apparent temp extra (°C)", 0, 10, 3, 1)
with s6:
    persistence_days = st.slider("Persistence days", 1, 7, 4, 1)

st.markdown("</div>", unsafe_allow_html=True)

try:
    baseline_df = make_ml_forecast(selected_city)
    stress_df = apply_stress_overrides(
        forecast_df=baseline_df,
        forced_temp_max=float(forced_temp_max),
        forced_temp_min=float(forced_temp_min),
        forced_apparent_extra=float(forced_apparent_extra),
        forced_humidity=float(forced_humidity),
        forced_wind=float(forced_wind),
        persistence_days=int(persistence_days),
        power_outage=power_outage,
    )
except Exception as exc:
    st.error(f"Stress Test nije dostupan: {exc}")
    st.stop()

baseline_summary = build_city_readiness_summary(selected_city, baseline_df)
stress_summary = build_city_readiness_summary(selected_city, stress_df)

stress_first_row_df = stress_df.sort_values("date").head(1).copy()
if "city" not in stress_first_row_df.columns:
    stress_first_row_df["city"] = selected_city

stress_escalation_df = predict_escalation_from_features(stress_first_row_df)
stress_escalation_row = stress_escalation_df.iloc[0]

v3_label = str(stress_escalation_row["escalation_label_72h"])
v3_probability = safe_float(stress_escalation_row["escalation_probability_72h"])

stress_consensus = build_stress_consensus(
    stress_first_row=stress_first_row_df.iloc[0],
    escalation_label=v3_label,
    escalation_probability=v3_probability,
    power_outage=power_outage,
    persistence_days=int(persistence_days),
)

render_app_sidebar(
    selected_city=selected_city,
    risk_level=str(stress_first_row_df.iloc[0].get("heuristic_risk_level", "N/A")),
    readiness_status=stress_summary["readiness_status"],
    escalation_label=v3_label,
    escalation_probability=v3_probability,
)

vulnerability_snapshot = get_city_vulnerability_snapshot(selected_city)
vulnerability_drivers = identify_vulnerability_drivers(vulnerability_snapshot)
vulnerability_recommendations = build_vulnerability_recommendations(vulnerability_snapshot)

impact_adjusted_priority = build_impact_adjusted_priority(
    next_7d_peak_score=safe_float(stress_summary["next_7d_peak_score"]),
    escalation_probability_72h=v3_probability,
    vulnerability_index=safe_float(vulnerability_snapshot["vulnerability_index"]),
)

priority_groups = identify_priority_groups(stress_summary, v3_label)
primary_impacts = identify_primary_impacts(stress_summary, v3_label)
operational_triggers = build_operational_triggers(stress_summary, v3_label)
sector_actions = build_sector_actions(stress_summary["next_7d_peak_level"])
escalation_plan = build_escalation_plan(stress_summary["next_7d_peak_level"])

dispatch_df = recommend_dispatch_resources(
    city=selected_city,
    escalation_label=v3_label,
    priority_groups=priority_groups,
    top_n=5,
)

top_dispatch_summary = (
    build_top_dispatch_summary(dispatch_df)
    if not dispatch_df.empty
    else "No dispatch resource recommendation available for this stress scenario."
)

xai_summary = explain_escalation_row(stress_first_row_df)

stress_brief = build_stress_brief(
    city=selected_city,
    baseline_summary=baseline_summary,
    stress_summary=stress_summary,
    stress_consensus=stress_consensus,
    vulnerability_snapshot=vulnerability_snapshot,
    impact_adjusted_priority=impact_adjusted_priority,
    top_dispatch_summary=top_dispatch_summary,
    xai_summary=xai_summary,
    forced_temp_max=float(forced_temp_max),
    forced_humidity=float(forced_humidity),
    forced_wind=float(forced_wind),
    persistence_days=int(persistence_days),
    power_outage=power_outage,
)

delta_peak = safe_float(stress_summary["next_7d_peak_score"]) - safe_float(baseline_summary["next_7d_peak_score"])
baseline_peak_date = pd.to_datetime(baseline_summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")
stress_peak_date = pd.to_datetime(stress_summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Stress readiness", stress_summary["readiness_status"], "Synthetic extreme mode")
with k2:
    metric_card("Stress peak", stress_summary["next_7d_peak_level"], f"{safe_float(stress_summary['next_7d_peak_score']):.1f}")
with k3:
    metric_card("v3 escalation", v3_label, f"{format_float(v3_probability, 2)}")
with k4:
    metric_card("Impact priority", format_float(impact_adjusted_priority, 1), vulnerability_snapshot["vulnerability_band"])
with k5:
    metric_card(
        "Operator review",
        "YES" if stress_consensus["operator_review_required"] else "NO",
        stress_consensus["confidence_level"],
    )

badge(stress_summary["readiness_status"], readiness_to_color(stress_summary["readiness_status"]))
badge(
    stress_consensus["consensus_status"],
    CONSENSUS_COLOR_MAP.get(stress_consensus["consensus_status"], "#64748b"),
)
badge(
    stress_consensus["confidence_level"],
    CONFIDENCE_COLOR_MAP.get(stress_consensus["confidence_level"], "#64748b"),
)

st.markdown(
    f"""
    <div class="warning-box">
        <b>Stress-test interpretation:</b> za grad <b>{selected_city}</b> sintetički scenarij podiže
        7-dnevni peak score za <b>{delta_peak:+.1f}</b> u odnosu na baseline.
        Aktivni readiness prelazi iz <b>{baseline_summary["readiness_status"]}</b> u
        <b>{stress_summary["readiness_status"]}</b>, dok je v3 escalation signal
        <b>{v3_label}</b> uz probability <b>{format_float(v3_probability, 2)}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="summary-card">
        Stress Test stranica nije samo vizualni eksperiment, nego <b>operativni robustness layer</b>.
        Ona pokazuje kako HeatSafe HR reagira kada se za <b>{selected_city}</b> umjetno uvedu ekstremniji
        ljetni uvjeti, višednevna perzistencija i eventualno infrastrukturno pogoršanje.
        U tom režimu platforma i dalje mora dati smislen readiness, dispatch, vulnerability i XAI output.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Stress overview",
        "Dispatch & response",
        "Reliability & XAI",
        "Stress report",
    ]
)

with tabs[0]:
    st.markdown("### Baseline vs stress timeline")

    baseline_plot = baseline_df[["date", "heuristic_risk_score"]].copy()
    baseline_plot["scenario"] = "Baseline"

    stress_plot = stress_df[["date", "heuristic_risk_score"]].copy()
    stress_plot["scenario"] = "Stress test"

    combined_plot = pd.concat([baseline_plot, stress_plot], ignore_index=True)

    fig = px.line(
        combined_plot,
        x="date",
        y="heuristic_risk_score",
        color="scenario",
        markers=True,
        title=f"Baseline vs stress-test risk trajectory — {selected_city}",
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Datum",
        yaxis_title="Risk score",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(range=[0, 100], gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        metric_card("Baseline readiness", baseline_summary["readiness_status"], f"Peak: {baseline_summary['next_7d_peak_level']}")
    with a2:
        metric_card("Stress readiness", stress_summary["readiness_status"], f"Peak: {stress_summary['next_7d_peak_level']}")
    with a3:
        metric_card("Peak delta", f"{delta_peak:+.1f}", f"{baseline_peak_date} → {stress_peak_date}")

    i1, i2, i3 = st.columns(3)
    with i1:
        render_list_card("Primary impacts", primary_impacts)
    with i2:
        render_list_card("Priority groups", priority_groups)
    with i3:
        render_list_card("Operational triggers", operational_triggers)

    st.markdown("### Sector actions under stress")
    sa1, sa2, sa3 = st.columns(3)
    with sa1:
        render_list_card("Za grad", sector_actions["city"])
    with sa2:
        render_list_card("Za javne službe", sector_actions["services"])
    with sa3:
        render_list_card("Za turizam", sector_actions["tourism"])

    stress_day_table = stress_df[
        [
            "date",
            "heuristic_risk_level",
            "heuristic_risk_score",
            "temp_max",
            "apparent_temp_max",
            "humidity_mean",
            "wind_speed_mean",
        ]
    ].copy()
    stress_day_table["date"] = pd.to_datetime(stress_day_table["date"]).dt.strftime("%d.%m.%Y.")
    st.dataframe(stress_day_table, use_container_width=True, hide_index=True)

with tabs[1]:
    st.markdown("### Dispatch & response layer")

    impact_band = impact_band_from_peak(stress_summary["next_7d_peak_level"], v3_label)

    d1, d2, d3 = st.columns(3)
    with d1:
        metric_card("Impact band", impact_band, "Stress severity")
    with d2:
        metric_card(
            "Top dispatch",
            dispatch_df.iloc[0]["resource_name"] if not dispatch_df.empty else "N/A",
            "Best operational route",
        )
    with d3:
        metric_card(
            "Dispatch score",
            format_float(dispatch_df.iloc[0]["dispatch_score"], 1) if not dispatch_df.empty else "N/A",
            "Top ranked point",
        )

    st.info(top_dispatch_summary)

    vd1, vd2 = st.columns(2)
    with vd1:
        render_list_card("Main vulnerability drivers", vulnerability_drivers)
    with vd2:
        render_list_card("Vulnerability-sensitive recommendations", vulnerability_recommendations)

    st.markdown("### Escalation logic under stress")
    e1, e2, e3 = st.columns(3)
    with e1:
        render_list_card("Što napraviti odmah", escalation_plan["immediately"])
    with e2:
        render_list_card("Što napraviti u 24h", escalation_plan["within_24h"])
    with e3:
        render_list_card("Što napraviti u 72h", escalation_plan["within_72h"])

    if dispatch_df.empty:
        st.info("Nema dispatch resource preporuka za ovaj grad.")
    else:
        dispatch_display_df = dispatch_df[
            [
                "dispatch_rank",
                "resource_name",
                "resource_type",
                "dispatch_score",
                "readiness_score",
                "capacity_availability_score",
                "proximity_score",
                "opening_score",
                "trust_score",
                "priority_fit_score",
                "nearest_critical_point",
                "nearest_critical_distance_km",
                "dispatch_reason",
            ]
        ].copy()
        st.dataframe(dispatch_display_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("### Reliability & XAI under stress")

    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Synthetic v1 signal", stress_consensus["v1_signal"], "Stress-adjusted risk")
    with r2:
        metric_card("v3 signal", stress_consensus["v3_signal"], format_float(v3_probability, 2))
    with r3:
        metric_card("Consensus", stress_consensus["consensus_status"], stress_consensus["confidence_level"])

    x1, x2 = st.columns(2)
    with x1:
        render_text_card(
            "Stress reliability signal",
            f"""
            <b>Consensus status:</b> {stress_consensus["consensus_status"]}<br>
            <b>Confidence level:</b> {stress_consensus["confidence_level"]}<br>
            <b>Operator review required:</b> {"Yes" if stress_consensus["operator_review_required"] else "No"}<br>
            <b>Uncertainty warning:</b> {stress_consensus["uncertainty_warning"]}
            """,
        )
    with x2:
        render_text_card(
            "Stress XAI summary",
            f"""
            <b>Method:</b> {xai_summary.get("method", "N/A")}<br>
            <b>Probability:</b> {format_float(xai_summary.get("probability"), 2)}<br>
            <b>Label:</b> {xai_summary.get("label", "N/A")}<br>
            <b>Explanation:</b> {xai_summary.get("explanation_text", "N/A")}
            """,
        )

    xp1, xp2 = st.columns(2)
    with xp1:
        render_list_card(
            "Top positive drivers",
            [
                f"{item['feature']} (contribution: {item['contribution']})"
                for item in xai_summary.get("top_positive_drivers", [])
            ] or ["Nema dostupnih pozitivnih drivera."],
        )
    with xp2:
        render_list_card(
            "Top protective drivers",
            [
                f"{item['feature']} (contribution: {item['contribution']})"
                for item in xai_summary.get("top_protective_drivers", [])
            ] or ["Nema dostupnih zaštitnih drivera."],
        )

    if stress_consensus["operator_review_required"]:
        st.error(stress_consensus["uncertainty_warning"])
    else:
        st.success("Stress scenario signal is usable without immediate major uncertainty warning.")

with tabs[3]:
    st.markdown("### Stress report")

    st.code(stress_brief, language="text")

    export_df = stress_df[
        [
            "date",
            "heuristic_risk_level",
            "heuristic_risk_score",
            "temp_max",
            "apparent_temp_max",
            "humidity_mean",
            "wind_speed_mean",
        ]
    ].copy()
    export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%d.%m.%Y.")

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "⬇ Download stress brief (.txt)",
            data=stress_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_stress_test_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_stress_brief_{selected_city}",
        )
    with b2:
        st.download_button(
            "⬇ Download stress timeline (.csv)",
            data=csv_bytes(export_df),
            file_name=f"heatsafe_hr_stress_test_timeline_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_stress_csv_{selected_city}",
        )

st.markdown(
    """
    <div class="note-box">
        <b>Why this page matters:</b> Stress Test dodaje pravi natjecateljski i istraživački “wow faktor”.
        On pokazuje da HeatSafe HR ne služi samo za pasivno promatranje forecasta, nego i za aktivno
        testiranje robusnosti sustava pod ekstremnim uvjetima, uz readiness, dispatch, reliability i XAI sloj.
    </div>
    """,
    unsafe_allow_html=True,
)