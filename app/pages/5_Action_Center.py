from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.decision_engine import (
    assess_event_risk,
    build_city_readiness_summary,
    build_escalation_plan,
    build_event_brief,
    build_executive_brief,
    build_scenario_comparison_brief,
    build_sector_actions,
    readiness_to_color,
    risk_to_color,
)
from src.escalation_engine import predict_escalation_from_features
from src.forecast_engine import make_ml_forecast
from src.impact_engine import (
    build_civil_protection_executive_brief,
    build_operational_triggers,
    identify_primary_impacts,
    identify_priority_groups,
    impact_band_from_peak,
)
from src.pdf_export_engine import generate_daily_briefing_pdf
from src.resource_recommender import recommend_resources
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

st.set_page_config(
    page_title="Action Center | HeatSafe HR",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}
ESCALATION_COLOR_MAP = {
    "Stable": "#64748b",
    "Watch": "#E6A700",
    "Likely escalation": "#C0392B",
}
VULNERABILITY_COLOR_MAP = {
    "Lower vulnerability": "#2E8B57",
    "Moderate vulnerability": "#E6A700",
    "High vulnerability": "#E67E22",
    "Very high vulnerability": "#C0392B",
}


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1360px;
        }

        .page-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
            border-radius: 22px;
            padding: 1.45rem 1.6rem 1.25rem 1.6rem;
            color: white;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 28px rgba(0,0,0,0.16);
        }

        .page-hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .page-hero-subtitle {
            font-size: 1rem;
            line-height: 1.62;
            opacity: 0.95;
            max-width: 1100px;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }

        .chip {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: white;
            font-size: 0.88rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.4rem 0 0.9rem 0;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            min-height: 116px;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.38rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
            margin-bottom: 0.18rem;
            word-break: break-word;
        }

        .metric-sub {
            font-size: 0.88rem;
            color: #64748b;
            line-height: 1.5;
        }

        .status-pill {
            display: inline-block;
            padding: 0.42rem 0.82rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.92rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .soft-panel {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            height: 100%;
        }

        .panel-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.75rem;
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

        .note-box {
            background: #eff6ff;
            border-left: 6px solid #2563eb;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            color: #0f172a;
            margin: 0.75rem 0 1rem 0;
            line-height: 1.65;
        }

        .warning-box {
            background: #fff7ed;
            border-left: 6px solid #ea580c;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            color: #7c2d12;
            margin: 0.75rem 0 1rem 0;
            line-height: 1.65;
        }

        .report-box {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            margin-top: 0.8rem;
        }

        .subsection-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.25rem 0 0.8rem 0;
        }

        .compact-card {
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            height: 100%;
        }

        .compact-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .compact-value {
            font-size: 1.22rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .compact-text {
            font-size: 0.92rem;
            color: #475569;
            line-height: 1.6;
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.72rem 1rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            box-shadow: 0 8px 20px rgba(29, 78, 216, 0.22);
            transition: all 0.2s ease-in-out;
        }

        div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(29, 78, 216, 0.30);
            background: linear-gradient(135deg, #1e293b 0%, #2563eb 100%);
            color: white;
        }

        div.stDownloadButton > button:focus:not(:active) {
            border: none;
            color: white;
        }

        div[data-baseweb="tab-list"] {
            gap: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        button[data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            font-weight: 700;
            padding: 0.58rem 0.95rem;
            margin-right: 0.25rem;
        }

        div[data-baseweb="tab-panel"] {
            padding-top: 1rem;
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


def driver_items(driver_list: list[dict]) -> list[str]:
    if not driver_list:
        return ["Nema dostupnih drivera za ovaj signal."]
    return [f"{item['feature']} (contribution: {item['contribution']})" for item in driver_list]


def to_display_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column]).dt.strftime("%d.%m.%Y.")
    return out


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_live_escalation_snapshot(city: str, forecast_df: pd.DataFrame) -> dict:
    if forecast_df.empty:
        return {
            "escalation_probability_72h": None,
            "escalation_flag_72h": None,
            "escalation_label_72h": None,
        }

    first_row = forecast_df.sort_values("date").head(1).copy()
    if "city" not in first_row.columns:
        first_row["city"] = city

    pred_df = predict_escalation_from_features(first_row)
    row = pred_df.iloc[0]

    return {
        "escalation_probability_72h": float(row["escalation_probability_72h"]),
        "escalation_flag_72h": int(row["escalation_flag_72h"]),
        "escalation_label_72h": str(row["escalation_label_72h"]),
    }


def build_timeline_figure(df: pd.DataFrame, city_name: str) -> go.Figure:
    plot_df = df.copy().sort_values("date").reset_index(drop=True)
    plot_df["date_str"] = pd.to_datetime(plot_df["date"]).dt.strftime("%d.%m.%Y.")

    colors = [RISK_COLOR_MAP.get(level, "#64748b") for level in plot_df["heuristic_risk_level"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["date_str"],
            y=plot_df["heuristic_risk_score"],
            marker_color=colors,
            text=plot_df["heuristic_risk_score"].round(0).astype(int).astype(str),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Risk level: %{customdata[0]}<br>"
                "Risk score: %{y:.1f}<br>"
                "ML label: %{customdata[1]}<br>"
                "ML confidence: %{customdata[2]:.2f}<br>"
                "Temp max: %{customdata[3]:.1f} °C<br>"
                "Apparent temp max: %{customdata[4]:.1f} °C<br>"
                "Humidity mean: %{customdata[5]:.1f} %<extra></extra>"
            ),
            customdata=plot_df[
                [
                    "heuristic_risk_level",
                    "ml_predicted_label",
                    "ml_prediction_confidence",
                    "temp_max",
                    "apparent_temp_max",
                    "humidity_mean",
                ]
            ].values,
            name="Projected risk score",
        )
    )

    for y, label, color in [
        (25, "Umjeren", "#E6A700"),
        (50, "Visok", "#E67E22"),
        (75, "Vrlo visok", "#C0392B"),
    ]:
        fig.add_hline(
            y=y,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            opacity=0.55,
            annotation_text=label,
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"Operational risk timeline — {city_name}",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        bargap=0.18,
    )

    fig.update_xaxes(title_text="Datum", showgrid=False, tickangle=0)
    fig.update_yaxes(
        title_text="Projected Heat Risk Score",
        range=[0, 100],
        showgrid=True,
        gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
    )
    return fig


inject_custom_css()

cities = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = cities.index(default_city) if default_city in cities else 0

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🚨 Action Center</div>
        <div class="page-hero-subtitle">
            Operativni command sloj za toplinski rizik. Ova stranica spaja forecast, readiness,
            72h escalation signal, vulnerability layer, impact-based forecasting, XAI,
            resource routing i export-ready briefing u jedinstven radni pogled za grad,
            službe i koordinacijske timove.
        </div>
        <div class="chip-row">
            <span class="chip">Decision Support</span>
            <span class="chip">Escalation</span>
            <span class="chip">Vulnerability</span>
            <span class="chip">XAI</span>
            <span class="chip">Routing</span>
            <span class="chip">PDF Briefing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Operational control panel</div>', unsafe_allow_html=True)

top1, top2 = st.columns([1, 1])
with top1:
    selected_city = st.selectbox("Odaberi grad", cities, index=default_index)
    st.session_state.selected_city = selected_city
with top2:
    scenario_enabled = st.toggle("Scenario mode", value=True)

if scenario_enabled:
    s1, s2, s3 = st.columns(3)
    with s1:
        temperature_delta = st.slider("Promjena temperature (°C)", -2, 12, 6, 1)
    with s2:
        humidity_delta = st.slider("Promjena vlage (%)", -20, 30, 10, 1)
    with s3:
        wind_delta = st.slider("Promjena vjetra (m/s)", -8, 5, -3, 1)
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

try:
    baseline_df = make_ml_forecast(selected_city)
    scenario_df = make_ml_forecast(
        selected_city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
except Exception as exc:
    st.error(f"Action Center nije dostupan: {exc}")
    st.stop()

active_df = scenario_df if scenario_enabled else baseline_df

summary = build_city_readiness_summary(selected_city, active_df)
actions = build_sector_actions(summary["next_7d_peak_level"])
escalation_plan = build_escalation_plan(summary["next_7d_peak_level"])
live_escalation = build_live_escalation_snapshot(selected_city, active_df)

xai_input_row = active_df.sort_values("date").head(1).copy()
if "city" not in xai_input_row.columns:
    xai_input_row["city"] = selected_city
xai_summary = explain_escalation_row(xai_input_row)

vulnerability_snapshot = get_city_vulnerability_snapshot(selected_city)
vulnerability_drivers = identify_vulnerability_drivers(vulnerability_snapshot)
vulnerability_recommendations = build_vulnerability_recommendations(vulnerability_snapshot)

impact_adjusted_priority = build_impact_adjusted_priority(
    next_7d_peak_score=float(summary["next_7d_peak_score"]),
    escalation_probability_72h=live_escalation["escalation_probability_72h"],
    vulnerability_index=float(vulnerability_snapshot["vulnerability_index"]),
)

recommended_resources_df = recommend_resources(
    city=selected_city,
    escalation_label=live_escalation["escalation_label_72h"] or "Stable",
    top_n=3,
)

impact_band = impact_band_from_peak(
    summary["next_7d_peak_level"],
    live_escalation["escalation_label_72h"],
)

primary_impacts = identify_primary_impacts(
    summary,
    live_escalation["escalation_label_72h"],
)

priority_groups = identify_priority_groups(
    summary,
    live_escalation["escalation_label_72h"],
)

dispatch_resources_df = recommend_dispatch_resources(
    city=selected_city,
    escalation_label=live_escalation["escalation_label_72h"] or "Stable",
    priority_groups=priority_groups,
    top_n=5,
)

top_dispatch_summary = build_top_dispatch_summary(dispatch_resources_df)

operational_triggers = build_operational_triggers(
    summary,
    live_escalation["escalation_label_72h"],
)

civil_protection_brief = build_civil_protection_executive_brief(
    city=selected_city,
    summary=summary,
    escalation_probability=live_escalation["escalation_probability_72h"],
    escalation_label=live_escalation["escalation_label_72h"],
    recommended_resources=recommended_resources_df,
    scenario_enabled=scenario_enabled,
    temperature_delta=temperature_delta,
    humidity_delta=humidity_delta,
    wind_delta=wind_delta,
)

civil_protection_brief = (
    civil_protection_brief
    + "\n\nVULNERABILITY LAYER\n"
    + f"- Vulnerability index: {vulnerability_snapshot['vulnerability_index']:.2f}\n"
    + f"- Vulnerability band: {vulnerability_snapshot['vulnerability_band']}\n"
    + "- Main drivers:\n"
    + "\n".join(f"  - {item}" for item in vulnerability_drivers)
    + "\n- Vulnerability-sensitive recommendations:\n"
    + "\n".join(f"  - {item}" for item in vulnerability_recommendations)
)

positive_driver_lines = (
    "\n".join(
        f"  - {item['feature']} ({item['contribution']})"
        for item in xai_summary["top_positive_drivers"]
    )
    if xai_summary["top_positive_drivers"]
    else "  - No dominant positive drivers detected"
)

protective_driver_lines = (
    "\n".join(
        f"  - {item['feature']} ({item['contribution']})"
        for item in xai_summary["top_protective_drivers"]
    )
    if xai_summary["top_protective_drivers"]
    else "  - No dominant protective drivers detected"
)

civil_protection_brief = (
    civil_protection_brief
    + "\n\nXAI / EXPLAINABLE AI LAYER\n"
    + f"- Method: {xai_summary['method']}\n"
    + f"- Probability: {xai_summary['probability']:.2f}\n"
    + f"- Label: {xai_summary['label']}\n"
    + "- Top positive drivers:\n"
    + positive_driver_lines
    + "\n- Top protective drivers:\n"
    + protective_driver_lines
    + f"\n- Summary: {xai_summary['explanation_text']}"
)

civil_protection_brief = (
    civil_protection_brief
    + "\n\nOPERATIONAL RESOURCE ROUTING\n"
    + f"- {top_dispatch_summary}\n"
)

executive_brief = build_executive_brief(
    selected_city,
    active_df,
    scenario_used=scenario_enabled,
)
scenario_brief = build_scenario_comparison_brief(
    selected_city,
    baseline_df,
    scenario_df,
    temperature_delta=temperature_delta,
    humidity_delta=humidity_delta,
    wind_delta=wind_delta,
)

pdf_reliability_snapshot = {
    "v1_signal": str(
        active_df.sort_values("date").iloc[0].get(
            "ml_predicted_label",
            active_df.sort_values("date").iloc[0].get("heuristic_risk_level", "N/A"),
        )
    ),
    "v2_signal": "N/A",
    "v3_signal": live_escalation["escalation_label_72h"],
    "confidence_level": "N/A",
    "reliability_score": 0.0,
    "operator_review_required": False,
    "uncertainty_warning": "Action Center PDF export mode.",
    "consensus_status": "Action Center mode",
}

public_alert_summary = (
    f"{selected_city}: readiness status is {summary['readiness_status']}. "
    f"Next 7d peak is {summary['next_7d_peak_level']} "
    f"({summary['next_7d_peak_score']:.1f}) on "
    f"{summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}. "
    f"72h escalation signal is {live_escalation['escalation_label_72h']} "
    f"with probability {live_escalation['escalation_probability_72h']:.2f}."
)

scenario_meta = (
    {
        "temperature_delta": temperature_delta,
        "humidity_delta": humidity_delta,
        "wind_delta": wind_delta,
    }
    if scenario_enabled
    else None
)

daily_briefing_pdf_bytes = generate_daily_briefing_pdf(
    city=selected_city,
    summary=summary,
    reliability_snapshot=pdf_reliability_snapshot,
    vulnerability_snapshot=vulnerability_snapshot,
    impact_adjusted_priority=impact_adjusted_priority,
    impact_band=impact_band,
    priority_groups=priority_groups,
    primary_impacts=primary_impacts,
    operational_triggers=operational_triggers,
    sector_actions=actions,
    xai_summary=xai_summary,
    top_dispatch_summary=top_dispatch_summary,
    public_alert_summary=public_alert_summary,
    timeline_df=active_df,
    scenario_enabled=scenario_enabled,
    scenario_meta=scenario_meta,
)

render_app_sidebar(
    selected_city=selected_city,
    risk_level=summary["next_24h_level"],
    readiness_status=summary["readiness_status"],
    escalation_label=live_escalation["escalation_label_72h"],
    escalation_probability=live_escalation["escalation_probability_72h"],
)

st.markdown(
    f"""
    <div class="note-box">
        <b>Operator framing:</b> Action Center pretvara forecast signal u konkretne odluke.
        Fokus nije samo na tome koliki je toplinski rizik, nego i na tome
        <b>koga pogađa, što treba napraviti, koji resurs prvi aktivirati i kako pripremiti komunikaciju</b>.
        Trenutno je za grad <b>{selected_city}</b> readiness status <b>{summary['readiness_status']}</b>,
        a 72h escalation signal iznosi <b>{live_escalation['escalation_label_72h']}</b>
        uz probability <b>{live_escalation['escalation_probability_72h']:.2f}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

if scenario_enabled:
    st.markdown(
        f"""
        <div class="warning-box">
            <b>Scenario mode active:</b> koristiš simulirani operativni scenarij s promjenama
            <b>ΔT {temperature_delta:+.1f} °C</b>, <b>ΔRH {humidity_delta:+.1f}%</b> i
            <b>ΔWind {wind_delta:+.1f} m/s</b>. Ovo je korisno za what-if planiranje,
            testiranje spremnosti i demonstraciju kako se signal mijenja pod težim uvjetima.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Current operational snapshot</div>', unsafe_allow_html=True)

current_risk_color = risk_to_color(summary["next_24h_level"])
current_readiness_color = readiness_to_color(summary["readiness_status"])
current_escalation_color = ESCALATION_COLOR_MAP.get(live_escalation["escalation_label_72h"], "#64748b")
current_vulnerability_color = VULNERABILITY_COLOR_MAP.get(vulnerability_snapshot["vulnerability_band"], "#64748b")

status_left, status_right = st.columns([1.2, 1])
with status_left:
    st.markdown(
        f"""
        <div class="compact-card">
            <div class="compact-title">Selected city command view</div>
            <div class="compact-value">{selected_city}</div>
            <div style="margin:0.45rem 0 0.75rem 0;">
                <span class="status-pill" style="background:{current_risk_color};">{summary["next_24h_level"]}</span>
                <span class="status-pill" style="background:{current_readiness_color};">{summary["readiness_status"]}</span>
                <span class="status-pill" style="background:{current_escalation_color};">{live_escalation["escalation_label_72h"]}</span>
                <span class="status-pill" style="background:{current_vulnerability_color};">{vulnerability_snapshot["vulnerability_band"]}</span>
            </div>
            <div class="compact-text">
                Peak within the next 7 days is expected on
                <b>{summary["next_7d_peak_date"].strftime("%d.%m.%Y.")}</b> with signal
                <b>{summary["next_7d_peak_level"]} ({summary["next_7d_peak_score"]:.1f})</b>.
                Impact-adjusted priority score is <b>{impact_adjusted_priority:.1f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_right:
    st.markdown(
        f"""
        <div class="compact-card">
            <div class="compact-title">Why this matters now</div>
            <div class="compact-text">
                Prioritetne skupine za trenutni signal uključuju <b>{len(priority_groups)}</b> grupa,
                a procijenjeni impact band je <b>{impact_band}</b>. Action Center ovdje spaja
                forecast, escalation, vulnerability, routing i XAI kako bi operater odmah imao
                i signal i preporučenu akciju, a ne samo meteorološki prikaz.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Next 24h risk", summary["next_24h_level"], "Immediate operational view")
with k2:
    metric_card("Next 24h score", f"{summary['next_24h_score']:.1f}", "Projected short-term score")
with k3:
    metric_card("72h escalation", live_escalation["escalation_label_72h"], f"{live_escalation['escalation_probability_72h']:.2f}")
with k4:
    metric_card("Impact priority", f"{impact_adjusted_priority:.1f}", "Heat + escalation + vulnerability")
with k5:
    metric_card("High-risk days", str(summary["high_risk_days"]), "Within next 7 days")

tabs = st.tabs(
    [
        "Action Center",
        "Event / Tourism Risk Check",
        "Executive Summary",
    ]
)

with tabs[0]:
    st.markdown('<div class="section-title">Operational forecast timeline</div>', unsafe_allow_html=True)

    peak_row = active_df.sort_values(
        ["heuristic_risk_score", "apparent_temp_max", "ml_prediction_confidence"],
        ascending=[False, False, False],
    ).iloc[0]

    trend_delta = float(active_df.iloc[-1]["heuristic_risk_score"] - active_df.iloc[0]["heuristic_risk_score"])

    g1, g2, g3 = st.columns(3)
    with g1:
        metric_card(
            "Peak day",
            pd.to_datetime(peak_row["date"]).strftime("%d.%m.%Y."),
            f"Risk level: {peak_row['heuristic_risk_level']}",
        )
    with g2:
        metric_card(
            "Peak score",
            f"{peak_row['heuristic_risk_score']:.1f}",
            f"ML label: {peak_row['ml_predicted_label']}",
        )
    with g3:
        metric_card(
            "ML confidence on peak",
            f"{peak_row['ml_prediction_confidence']:.2f}",
            f"Trend delta (7d): {trend_delta:+.1f}",
        )

    fig = build_timeline_figure(active_df, selected_city)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Sector recommendations</div>', unsafe_allow_html=True)
    rec1, rec2, rec3 = st.columns(3)
    with rec1:
        render_list_card("Preporuke za grad", actions["city"])
    with rec2:
        render_list_card("Preporuke za javne službe", actions["services"])
    with rec3:
        render_list_card("Preporuke za turizam", actions["tourism"])

    st.markdown('<div class="section-title">Escalation plan</div>', unsafe_allow_html=True)
    esc1, esc2, esc3 = st.columns(3)
    with esc1:
        render_list_card("Što napraviti odmah", escalation_plan["immediately"])
    with esc2:
        render_list_card("Što napraviti u 24h", escalation_plan["within_24h"])
    with esc3:
        render_list_card("Što napraviti u 72h", escalation_plan["within_72h"])

    st.markdown('<div class="section-title">Impact-based forecasting</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        metric_card("Impact band", impact_band, "Operational severity")
    with i2:
        metric_card(
            "72h escalation",
            live_escalation["escalation_label_72h"],
            f"{live_escalation['escalation_probability_72h']:.2f}"
            if live_escalation["escalation_probability_72h"] is not None
            else "N/A",
        )
    with i3:
        metric_card("Priority groups", str(len(priority_groups)), "Groups requiring attention")

    p1, p2, p3 = st.columns(3)
    with p1:
        render_list_card("Primary impacts", primary_impacts)
    with p2:
        render_list_card("Priority groups", priority_groups)
    with p3:
        render_list_card("Operational triggers", operational_triggers)

    st.markdown('<div class="section-title">Socio-economic vulnerability layer</div>', unsafe_allow_html=True)
    vv1, vv2, vv3 = st.columns(3)
    with vv1:
        metric_card(
            "Vulnerability index",
            f"{vulnerability_snapshot['vulnerability_index']:.1f}",
            "City-level profile",
        )
    with vv2:
        metric_card(
            "Vulnerability band",
            vulnerability_snapshot["vulnerability_band"],
            "Human-impact context",
        )
    with vv3:
        metric_card(
            "Impact-adjusted priority",
            f"{impact_adjusted_priority:.1f}",
            "Heat + escalation + vulnerability",
        )

    vd1, vd2 = st.columns(2)
    with vd1:
        render_list_card("Main vulnerability drivers", vulnerability_drivers)
    with vd2:
        render_list_card("Vulnerability-sensitive recommendations", vulnerability_recommendations)

    st.markdown('<div class="section-title">Explainable AI for v3 escalation signal</div>', unsafe_allow_html=True)
    x1, x2, x3 = st.columns(3)
    with x1:
        metric_card("XAI method", str(xai_summary["method"]), "Local explanation engine")
    with x2:
        metric_card(
            "V3 probability",
            f"{xai_summary['probability']:.2f}" if xai_summary["probability"] is not None else "N/A",
            "72h escalation probability",
        )
    with x3:
        metric_card("V3 label", str(xai_summary["label"]), "Model signal")

    xx1, xx2 = st.columns(2)
    with xx1:
        render_list_card("Top positive drivers", driver_items(xai_summary["top_positive_drivers"]))
    with xx2:
        render_list_card("Top protective drivers", driver_items(xai_summary["top_protective_drivers"]))

    st.info(xai_summary["explanation_text"])

    st.markdown('<div class="section-title">Operational resource routing</div>', unsafe_allow_html=True)

    if dispatch_resources_df.empty:
        st.info("Nema dostupnih dispatch resource preporuka za ovaj grad.")
    else:
        top_dispatch_row = dispatch_resources_df.iloc[0]

        dr1, dr2, dr3 = st.columns(3)
        with dr1:
            metric_card(
                "Top dispatch resource",
                str(top_dispatch_row["resource_name"]),
                str(top_dispatch_row["resource_type"]),
            )
        with dr2:
            metric_card(
                "Dispatch score",
                f"{top_dispatch_row['dispatch_score']:.1f}",
                "Operational routing score",
            )
        with dr3:
            distance_sub = (
                f"{top_dispatch_row['nearest_critical_distance_km']:.2f} km"
                if pd.notna(top_dispatch_row["nearest_critical_distance_km"])
                else "N/A"
            )
            metric_card(
                "Nearest critical point",
                str(top_dispatch_row["nearest_critical_point"]),
                distance_sub,
            )

        st.info(top_dispatch_summary)

        routing_display_df = dispatch_resources_df[
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

        st.dataframe(routing_display_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Recommended resources for current escalation signal</div>', unsafe_allow_html=True)
    if recommended_resources_df.empty:
        st.info("Nema preporučenih resource točaka za ovaj grad.")
    else:
        resource_columns = st.columns(min(3, len(recommended_resources_df)))
        for i, (_, row) in enumerate(recommended_resources_df.iterrows()):
            with resource_columns[i]:
                render_text_card(
                    row.get("resource_name", "Unknown"),
                    f"""
                    <b>Tip:</b> {row.get('resource_type', 'N/A')}<br>
                    <b>Adresa:</b> {row.get('address', 'N/A')}<br>
                    <b>Radno vrijeme:</b> {row.get('hours_weekday', 'N/A')}<br>
                    <b>Verified:</b> {row.get('verified_status', 'N/A')}<br>
                    <b>Water:</b> {row.get('water_available', 'N/A')}<br>
                    <b>Indoor cooling:</b> {row.get('indoor_cooling', 'N/A')}
                    """,
                )

    st.markdown('<div class="section-title">Executive summary for civil protection</div>', unsafe_allow_html=True)
    st.code(civil_protection_brief, language="text")

    st.markdown('<div class="section-title">Operativni pregled po danima</div>', unsafe_allow_html=True)
    timeline_df = active_df[
        [
            "date",
            "ml_predicted_label",
            "ml_prediction_confidence",
            "heuristic_risk_level",
            "heuristic_risk_score",
            "temp_max",
            "apparent_temp_max",
        ]
    ].copy()
    timeline_df["vulnerability_index"] = vulnerability_snapshot["vulnerability_index"]
    timeline_df["vulnerability_band"] = vulnerability_snapshot["vulnerability_band"]
    timeline_df["impact_adjusted_priority"] = impact_adjusted_priority
    day_table = to_display_date(timeline_df)
    st.dataframe(day_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Report export")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.download_button(
            "⬇ Download executive brief (.txt)",
            data=executive_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_executive_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_exec_action_{selected_city}",
        )
    with d2:
        st.download_button(
            "⬇ Download operational table (.csv)",
            data=csv_bytes(day_table),
            file_name=f"heatsafe_hr_operational_table_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_operational_csv_{selected_city}",
        )
    with d3:
        st.download_button(
            "⬇ Download scenario brief (.txt)",
            data=scenario_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_scenario_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_scenario_action_{selected_city}",
        )
    with d4:
        st.download_button(
            "⬇ Download civil protection brief (.txt)",
            data=civil_protection_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_civil_protection_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_civil_protection_brief_{selected_city}",
        )
    with d5:
        st.download_button(
            "⬇ Download daily briefing (.pdf)",
            data=daily_briefing_pdf_bytes,
            file_name=f"heatsafe_hr_daily_briefing_{selected_city}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_daily_briefing_pdf_{selected_city}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">Event / Tourism Risk Check</div>', unsafe_allow_html=True)

    day_options = active_df["date"].dt.strftime("%d.%m.%Y.").tolist()
    date_map = dict(zip(day_options, active_df["date"]))

    event_name = st.text_input("Naziv događaja", value="Summer Evening Event")

    e1, e2 = st.columns(2)
    with e1:
        selected_day_str = st.selectbox("Datum događaja", day_options)
    with e2:
        time_slot = st.selectbox(
            "Vrijeme događaja",
            ["Morning", "Midday / Afternoon", "Evening"],
            index=1,
        )

    e3, e4 = st.columns(2)
    with e3:
        event_type = st.selectbox(
            "Tip događaja",
            [
                "Outdoor concert",
                "Sports event",
                "Walking tour",
                "Festival / fair",
                "Beach / outdoor leisure",
                "City event / ceremony",
                "General tourism activity",
            ],
            index=2,
        )
    with e4:
        attendees = st.number_input(
            "Broj sudionika / gostiju",
            min_value=1,
            max_value=50000,
            value=250,
            step=50,
        )

    vulnerable_groups = st.toggle("Uključene osjetljive skupine", value=True)

    selected_date = date_map[selected_day_str]
    selected_row = active_df[active_df["date"] == selected_date].iloc[0]

    event_assessment = assess_event_risk(
        selected_row,
        event_type=event_type,
        attendees=int(attendees),
        time_slot=time_slot,
        vulnerable_groups=vulnerable_groups,
    )

    event_brief = build_event_brief(
        city_name=selected_city,
        event_name=event_name,
        event_date_str=selected_day_str,
        event_type=event_type,
        attendees=int(attendees),
        time_slot=time_slot,
        vulnerable_groups=vulnerable_groups,
        forecast_row=selected_row,
        assessment=event_assessment,
    )

    st.markdown(
        f"""
        <div class="note-box">
            <b>Event framing:</b> ova procjena ne gleda samo temperaturu, nego spaja forecast signal,
            broj sudionika, vrijeme održavanja i prisutnost osjetljivih skupina kako bi dao
            operativnu preporuku za prilagodbu ili provedbu događaja.
        </div>
        """,
        unsafe_allow_html=True,
    )

    l1, l2 = st.columns([1, 1])
    with l1:
        st.markdown("### Rezultat procjene događaja")
        x1, x2, x3 = st.columns(3)
        with x1:
            metric_card("Event risk score", f"{event_assessment.event_score:.1f}")
        with x2:
            metric_card("Event risk level", event_assessment.event_level)
        with x3:
            metric_card("Readiness status", event_assessment.readiness_status)

        event_readiness_color = readiness_to_color(event_assessment.readiness_status)
        badge(event_assessment.recommendation, event_readiness_color)

    with l2:
        render_list_card("Action items", event_assessment.action_items)

    st.markdown("### Signal iza procjene")
    signal_df = pd.DataFrame(
        [
            {"Metric": "Active risk level", "Value": selected_row["heuristic_risk_level"]},
            {"Metric": "Active risk score", "Value": f"{selected_row['heuristic_risk_score']:.1f}"},
            {"Metric": "ML label", "Value": selected_row["ml_predicted_label"]},
            {"Metric": "ML confidence", "Value": f"{selected_row['ml_prediction_confidence']:.2f}"},
            {"Metric": "Apparent temp max", "Value": f"{selected_row['apparent_temp_max']:.1f} °C"},
            {"Metric": "Max temperatura", "Value": f"{selected_row['temp_max']:.1f} °C"},
        ]
    )
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Event export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download event brief (.txt)",
            data=event_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_event_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_event_brief_{selected_city}_{selected_day_str}_{event_type}",
        )
    with c2:
        event_table = pd.DataFrame(
            [
                {
                    "event_name": event_name,
                    "city": selected_city,
                    "date": selected_day_str,
                    "event_type": event_type,
                    "attendees": int(attendees),
                    "time_slot": time_slot,
                    "vulnerable_groups": vulnerable_groups,
                    "event_score": event_assessment.event_score,
                    "event_level": event_assessment.event_level,
                    "readiness_status": event_assessment.readiness_status,
                    "recommendation": event_assessment.recommendation,
                }
            ]
        )
        st.download_button(
            "⬇ Download event summary (.csv)",
            data=csv_bytes(event_table),
            file_name=f"heatsafe_hr_event_summary_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_event_csv_{selected_city}_{selected_day_str}_{event_type}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

    top_exec_left, top_exec_right = st.columns(2)
    with top_exec_left:
        peak_signal_color = risk_to_color(summary["next_7d_peak_level"])
        render_text_card(
            "Peak executive signal",
            f"""
            <div style="margin-bottom:0.6rem;">
                <span class="status-pill" style="background:{peak_signal_color};">
                    {summary["next_7d_peak_level"]}
                </span>
            </div>
            <b>Peak date:</b> {summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}<br>
            <b>Peak score:</b> {summary['next_7d_peak_score']:.1f}<br>
            <b>Next 24h ML confidence:</b> {summary['next_24h_confidence']:.2f}
            """,
        )

    with top_exec_right:
        readiness_color = readiness_to_color(summary["readiness_status"])
        render_text_card(
            "Readiness summary",
            f"""
            <div style="margin-bottom:0.6rem;">
                <span class="status-pill" style="background:{readiness_color};">
                    {summary["readiness_status"]}
                </span>
            </div>
            <b>Next 24h risk:</b> {summary['next_24h_level']}<br>
            <b>Next 72h peak:</b> {summary['next_72h_peak_level']}<br>
            <b>High-risk days (7d):</b> {summary['high_risk_days']}
            """,
        )

    st.markdown("#### Executive brief")
    st.code(executive_brief, language="text")

    st.markdown("#### Scenario comparison brief")
    st.code(scenario_brief, language="text")

    st.markdown("#### Civil protection executive summary")
    st.code(civil_protection_brief, language="text")

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Brief export")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "⬇ Download executive brief (.txt)",
            data=executive_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_executive_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_exec_summary_{selected_city}",
        )
    with c2:
        st.download_button(
            "⬇ Download scenario brief (.txt)",
            data=scenario_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_scenario_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_scenario_summary_{selected_city}",
        )
    with c3:
        st.download_button(
            "⬇ Download civil protection brief (.txt)",
            data=civil_protection_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_civil_protection_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_civil_summary_{selected_city}",
        )
    with c4:
        st.download_button(
            "⬇ Download daily briefing (.pdf)",
            data=daily_briefing_pdf_bytes,
            file_name=f"heatsafe_hr_daily_briefing_{selected_city}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_daily_briefing_pdf_summary_{selected_city}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.success(
        """
        HeatSafe HR ovdje radi kao alat za odlučivanje:
        ne prikazuje samo prognozu, nego daje status pripravnosti, preporuke,
        impact-based forecasting, vulnerability layer, XAI i operativni brief
        za grad, javne službe i turistički sektor.
        """
    )