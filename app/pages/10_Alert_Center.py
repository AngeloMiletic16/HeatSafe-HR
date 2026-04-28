from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alert_engine import (
    ALERT_COLOR_MAP,
    append_alert_history,
    build_alert_package,
    build_operator_row,
    csv_bytes,
    get_alert_level,
    load_alert_history,
)
from src.communication_engine import build_alert_communication_package
from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary
from src.escalation_engine import predict_escalation_from_features
from src.forecast_engine import make_ml_forecast
from src.sidebar import render_app_sidebar

st.set_page_config(
    page_title="Alert Center | HeatSafe HR",
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

ALERT_HISTORY_PATH = PROJECT_ROOT / "data" / "alerts" / "alert_history.csv"

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

ESCALATION_COLOR_MAP = {
    "Stable": "#64748b",
    "Watch": "#e6a700",
    "Likely escalation": "#c0392b",
}


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1380px;
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
            font-size: 2.08rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .page-hero-subtitle {
            font-size: 1rem;
            line-height: 1.6;
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
            font-size: 1.38rem;
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
            min-height: 114px;
        }

        .metric-label {
            font-size: 0.82rem;
            color: #64748b;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            margin-bottom: 0.18rem;
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
            margin-bottom: 0.75rem;
        }

        .panel-text {
            color: #334155;
            line-height: 1.68;
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

        .status-pill {
            display: inline-block;
            padding: 0.36rem 0.76rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.86rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
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

        .report-box {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            margin-top: 0.8rem;
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

        div[data-baseweb="tab-list"] {
            gap: 1rem;
            margin-top: 0.9rem;
            margin-bottom: 0.9rem;
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


def format_date(value: Any) -> str:
    try:
        return pd.to_datetime(value).strftime("%d.%m.%Y.")
    except Exception:
        return str(value)


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


def pill(text: str, color: str) -> str:
    return f'<span class="status-pill" style="background:{color};">{text}</span>'


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


def build_communication_payload(
    city: str,
    summary: dict[str, Any],
    selected_row: pd.Series,
    forecast_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Normalizira payload za communication layer tako da svi tekstovi koriste
    iste readiness / peak / severity / escalation vrijednosti kao ostatak Alert Centera.
    """
    first_date = None
    if not forecast_df.empty and "date" in forecast_df.columns:
        first_date = pd.to_datetime(forecast_df.sort_values("date").iloc[0]["date"])

    target_audience_raw = selected_row.get("target_audience", "")
    if isinstance(target_audience_raw, str):
        target_audience_list = [x.strip() for x in target_audience_raw.split(",") if x.strip()]
    elif isinstance(target_audience_raw, list):
        target_audience_list = target_audience_raw
    else:
        target_audience_list = []

    next_24h_level = str(summary.get("next_24h_level", selected_row.get("peak", "N/A")))
    next_24h_score = safe_float(summary.get("next_24h_score", selected_row.get("peak_score", 0.0)))
    next_7d_peak_level = str(summary.get("next_7d_peak_level", selected_row.get("peak", "N/A")))
    next_7d_peak_score = safe_float(summary.get("next_7d_peak_score", selected_row.get("peak_score", 0.0)))
    next_7d_peak_date = summary.get("next_7d_peak_date", selected_row.get("peak_date"))

    payload = {
        # Core context
        "city": city,
        "date": first_date,
        "readiness": str(summary.get("readiness_status", selected_row.get("readiness", "N/A"))),
        "readiness_status": str(summary.get("readiness_status", selected_row.get("readiness", "N/A"))),
        "alert_severity": str(selected_row.get("alert_severity", "Monitoring Notice")),
        "alert_issued": str(selected_row.get("alert_issued", "No")),
        "alert_issued_bool": str(selected_row.get("alert_issued", "No")).strip().lower() == "yes",
        "target_audience": target_audience_raw,
        "target_audience_list": target_audience_list,
        "operator_summary": str(selected_row.get("operator_summary", "")),

        # Next 24h aliases
        "next_24h_level": next_24h_level,
        "next_24h_risk": next_24h_level,
        "next_24h_score": next_24h_score,

        # Peak aliases
        "peak": next_7d_peak_level,
        "peak_level": next_7d_peak_level,
        "peak_score": next_7d_peak_score,
        "peak_date": format_date(next_7d_peak_date),
        "next_7d_peak_level": next_7d_peak_level,
        "next_7d_peak_score": next_7d_peak_score,
        "next_7d_peak_date": next_7d_peak_date,
        "high_risk_days": int(summary.get("high_risk_days", selected_row.get("high_risk_days", 0))),

        # Escalation
        "escalation_probability_72h": safe_float(selected_row.get("escalation_probability_72h")),
        "escalation_label_72h": str(selected_row.get("escalation_label_72h", "Stable")),
    }

    return payload


@st.cache_data(ttl=1800)
def load_city_snapshot(
    city: str,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
):
    forecast_df = make_ml_forecast(
        city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )

    if "city" not in forecast_df.columns:
        forecast_df["city"] = city

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    summary = build_city_readiness_summary(city, forecast_df)
    return forecast_df, summary


inject_custom_css()

default_focus_city = st.session_state.get("selected_city", DEFAULT_CITY)
if default_focus_city not in CITIES:
    default_focus_city = CITIES[0]

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🚨 Alert Center</div>
        <div class="page-hero-subtitle">
            Operator-focused modul za izdavanje, pregled i distribuciju alerta. Alert Center spaja
            city snapshot, severity rules, 72h escalation signal i komunikacijski layer kako bi
            sustav iz forecasta prešao u konkretno upozorenje, dokumentaciju i javnu poruku.
        </div>
        <div class="chip-row">
            <span class="chip">Alerting</span>
            <span class="chip">Severity Rules</span>
            <span class="chip">Communication Layer</span>
            <span class="chip">Escalation</span>
            <span class="chip">History Log</span>
            <span class="chip">Operator Workflow</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Alert control panel</div>', unsafe_allow_html=True)

control_left, control_right = st.columns([1, 1])
with control_left:
    selected_city = st.selectbox(
        "Focus city",
        CITIES,
        index=CITIES.index(default_focus_city) if default_focus_city in CITIES else 0,
    )
    st.session_state.selected_city = selected_city

with control_right:
    scenario_enabled = st.toggle("Scenario mode za alerting", value=True)

if scenario_enabled:
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature_delta = st.slider("Promjena temperature (°C)", -2, 12, 6, 1)
    with c2:
        humidity_delta = st.slider("Promjena vlage (%)", -20, 30, 10, 1)
    with c3:
        wind_delta = st.slider("Promjena vjetra (m/s)", -8, 5, -3, 1)
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

rows = []
packages: dict[str, str] = {}

for city in CITIES:
    forecast_df, summary = load_city_snapshot(city, temperature_delta, humidity_delta, wind_delta)

    escalation = build_live_escalation_snapshot(city, forecast_df)

    alert = get_alert_level(
        summary,
        escalation_probability=escalation["escalation_probability_72h"],
        escalation_label=escalation["escalation_label_72h"],
    )

    row = build_operator_row(
        city=city,
        summary=summary,
        alert=alert,
        scenario_enabled=scenario_enabled,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
        escalation_probability=escalation["escalation_probability_72h"],
        escalation_label=escalation["escalation_label_72h"],
    )
    rows.append(row)

    packages[city] = build_alert_package(
        city=city,
        summary=summary,
        alert=alert,
        scenario_enabled=scenario_enabled,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
        escalation_probability=escalation["escalation_probability_72h"],
        escalation_label=escalation["escalation_label_72h"],
    )

snapshot_df = pd.DataFrame(rows).sort_values(
    ["alert_issued", "peak_score", "high_risk_days"],
    ascending=[False, False, False],
).reset_index(drop=True)

if snapshot_df.empty:
    st.warning("Nema dostupnih alert podataka za prikaz.")
    st.stop()

if selected_city in snapshot_df["city"].tolist():
    focus_row = snapshot_df[snapshot_df["city"] == selected_city].iloc[0]
else:
    focus_row = snapshot_df.iloc[0]
    selected_city = str(focus_row["city"])
    st.session_state.selected_city = selected_city

render_app_sidebar(
    selected_city=selected_city,
    risk_level=str(focus_row["peak"]),
    readiness_status=str(focus_row["readiness"]),
    escalation_label=str(focus_row["escalation_label_72h"]),
    escalation_probability=safe_float(focus_row["escalation_probability_72h"]),
)

issued_count = int((snapshot_df["alert_issued"] == "Yes").sum())
warning_plus = int(snapshot_df["alert_severity"].isin(["Heat Warning", "Critical Alert"]).sum())
critical_count = int((snapshot_df["alert_severity"] == "Critical Alert").sum())
top_city = str(snapshot_df.iloc[0]["city"])
likely_escalation_count = int((snapshot_df["escalation_label_72h"] == "Likely escalation").sum())

st.markdown(
    f"""
    <div class="note-box">
        <b>Operator framing:</b> Alert Center ne prikazuje samo tko ima visok score, nego i
        <b>kojem gradu treba poslati poruku, koju razinu upozorenja izdati i kako to prevesti u komunikaciju</b>.
        Trenutni focus city je <b>{selected_city}</b> uz severity <b>{focus_row["alert_severity"]}</b>,
        readiness <b>{focus_row["readiness"]}</b> i 72h escalation signal
        <b>{focus_row["escalation_label_72h"]}</b> ({safe_float(focus_row["escalation_probability_72h"]):.2f}).
    </div>
    """,
    unsafe_allow_html=True,
)

if scenario_enabled:
    st.markdown(
        f"""
        <div class="warning-box">
            <b>Scenario mode active:</b> alert snapshot je generiran pod uvjetima
            <b>ΔT {temperature_delta:+.1f} °C</b>, <b>ΔRH {humidity_delta:+.1f}%</b> i
            <b>ΔWind {wind_delta:+.1f} m/s</b>. Ovo je korisno za stress-testiranje alert logike,
            javne komunikacije i operator preparedness scenarija.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Current national alert snapshot</div>', unsafe_allow_html=True)

summary_left, summary_right = st.columns([1.15, 1])

with summary_left:
    focus_alert_color = ALERT_COLOR_MAP.get(str(focus_row["alert_severity"]), "#64748b")
    focus_escalation_color = ESCALATION_COLOR_MAP.get(str(focus_row["escalation_label_72h"]), "#64748b")

    st.markdown(
        f"""
        <div class="compact-card">
            <div class="compact-title">Focus city alert posture</div>
            <div class="compact-value">{selected_city}</div>
            <div style="margin:0.45rem 0 0.75rem 0;">
                <span class="status-pill" style="background:{focus_alert_color};">{focus_row["alert_severity"]}</span>
                <span class="status-pill" style="background:{focus_escalation_color};">{focus_row["escalation_label_72h"]}</span>
            </div>
            <div class="compact-text">
                <b>Peak:</b> {focus_row["peak"]} ({safe_float(focus_row["peak_score"]):.1f}) on {focus_row["peak_date"]}<br>
                <b>Readiness:</b> {focus_row["readiness"]}<br>
                <b>Target audience:</b> {focus_row["target_audience"]}<br>
                <b>Alert issued:</b> {focus_row["alert_issued"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_right:
    st.markdown(
        f"""
        <div class="compact-card">
            <div class="compact-title">Top operator priority</div>
            <div class="compact-value">{top_city}</div>
            <div class="compact-text">
                Sustav trenutno vidi <b>{issued_count}</b> izdanih alerta, od čega je
                <b>{warning_plus}</b> na razini Heat Warning ili više, dok je
                <b>{critical_count}</b> u Critical Alert režimu.
                V3 early-warning sloj trenutačno označava <b>{likely_escalation_count}</b>
                gradova kao Likely escalation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Alerts issued", str(issued_count), "Current snapshot")
with m2:
    metric_card("Likely escalation", str(likely_escalation_count), "V3 early-warning signal")
with m3:
    metric_card("Warning or higher", str(warning_plus), "Heat Warning + Critical")
with m4:
    metric_card("Critical alerts", str(critical_count), "Highest severity")
with m5:
    metric_card("Top operator priority", top_city, "Most urgent city")

tabs = st.tabs(
    [
        "Current snapshot",
        "Alert history",
        "Alert packages",
        "Communication layer",
    ]
)

with tabs[0]:
    st.markdown('<div class="section-title">Current operator snapshot</div>', unsafe_allow_html=True)

    fig = px.bar(
        snapshot_df,
        x="city",
        y="peak_score",
        color="alert_severity",
        color_discrete_map=ALERT_COLOR_MAP,
        text="alert_severity",
        title="Current city alert severity snapshot",
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Grad",
        yaxis_title="Peak score",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig, use_container_width=True)

    display_df = snapshot_df[
        [
            "date",
            "city",
            "readiness",
            "peak",
            "peak_score",
            "peak_date",
            "escalation_probability_72h",
            "escalation_label_72h",
            "alert_severity",
            "alert_issued",
            "target_audience",
            "operator_summary",
        ]
    ].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Snapshot actions")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save current alert snapshot to history", use_container_width=True):
            append_alert_history(snapshot_df, ALERT_HISTORY_PATH)
            st.success("Alert snapshot spremljen u history log.")
    with c2:
        st.download_button(
            "⬇ Download current alert snapshot (.csv)",
            data=csv_bytes(snapshot_df),
            file_name="heatsafe_hr_current_alert_snapshot.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_current_alert_snapshot",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">Alert history</div>', unsafe_allow_html=True)

    history_df = load_alert_history(ALERT_HISTORY_PATH)

    if history_df.empty:
        st.info("Alert history je trenutno prazan. Prvo spremi current snapshot.")
    else:
        h1, h2 = st.columns(2)
        with h1:
            city_filter = st.multiselect(
                "Filtriraj gradove",
                options=sorted(history_df["city"].dropna().unique().tolist()),
                default=sorted(history_df["city"].dropna().unique().tolist()),
            )
        with h2:
            severity_filter = st.multiselect(
                "Filtriraj severity",
                options=sorted(history_df["alert_severity"].dropna().unique().tolist()),
                default=sorted(history_df["alert_severity"].dropna().unique().tolist()),
            )

        filtered_history = history_df[
            history_df["city"].isin(city_filter) & history_df["alert_severity"].isin(severity_filter)
        ].copy()

        st.markdown(
            """
            <div class="note-box">
                <b>History framing:</b> Alert history služi za audit trail i povratnu analizu.
                Omogućuje pregled kada je alert logika aktivirana, za koje gradove i pod kojim severity pravilima.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(filtered_history, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download alert history (.csv)",
            data=csv_bytes(filtered_history),
            file_name="heatsafe_hr_alert_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_alert_history",
        )

with tabs[2]:
    st.markdown('<div class="section-title">Exportable alert package</div>', unsafe_allow_html=True)

    package_text = packages[selected_city]
    selected_row = snapshot_df[snapshot_df["city"] == selected_city].iloc[0]

    severity_badge = pill(
        selected_row["alert_severity"],
        ALERT_COLOR_MAP.get(selected_row["alert_severity"], "#64748b"),
    )
    st.markdown(severity_badge, unsafe_allow_html=True)

    if pd.notna(selected_row.get("escalation_label_72h")):
        escalation_badge = pill(
            selected_row["escalation_label_72h"],
            ESCALATION_COLOR_MAP.get(selected_row["escalation_label_72h"], "#64748b"),
        )
        st.markdown(escalation_badge, unsafe_allow_html=True)

    info_left, info_right = st.columns(2)
    with info_left:
        render_text_card(
            "Package context",
            f"""
            <b>City:</b> {selected_city}<br>
            <b>Severity:</b> {selected_row["alert_severity"]}<br>
            <b>Readiness:</b> {selected_row["readiness"]}<br>
            <b>Alert issued:</b> {selected_row["alert_issued"]}<br>
            <b>Target audience:</b> {selected_row["target_audience"]}
            """,
        )
    with info_right:
        render_text_card(
            "Operator summary",
            f"""
            {selected_row["operator_summary"]}
            """,
        )

    st.code(package_text, language="text")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download alert package (.txt)",
            data=package_text.encode("utf-8"),
            file_name=f"heatsafe_hr_alert_package_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_alert_package_{selected_city}",
        )
    with c2:
        st.download_button(
            "⬇ Download operator row (.csv)",
            data=csv_bytes(snapshot_df[snapshot_df["city"] == selected_city]),
            file_name=f"heatsafe_hr_operator_row_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_operator_row_{selected_city}",
        )

with tabs[3]:
    st.markdown('<div class="section-title">Smart alerting communication layer</div>', unsafe_allow_html=True)

    if snapshot_df.empty:
        st.info("Nema alert podataka za generiranje komunikacijskih poruka.")
    else:
        selected_alert_row = snapshot_df[snapshot_df["city"] == selected_city].iloc[0]
        selected_forecast_df, selected_summary = load_city_snapshot(
            selected_city,
            temperature_delta,
            humidity_delta,
            wind_delta,
        )

        communication_payload = build_communication_payload(
            city=selected_city,
            summary=selected_summary,
            selected_row=selected_alert_row,
            forecast_df=selected_forecast_df,
        )

        comm_package = build_alert_communication_package(communication_payload)

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Alert severity", selected_alert_row["alert_severity"], "Public communication level")
        with c2:
            metric_card("Readiness", communication_payload["readiness"], "Operational posture")
        with c3:
            metric_card(
                "72h escalation",
                communication_payload["escalation_label_72h"],
                f"{safe_float(communication_payload['escalation_probability_72h']):.2f}",
            )

        st.markdown(
            f"""
            <div class="note-box">
                <b>Communication framing:</b> Ovdje HeatSafe HR prevodi alert signal u poruke za javnost,
                turiste, medije i operatere. Ovaj komunikacijski paket je sada sinkroniziran s aktivnim
                snapshotom za grad <b>{selected_city}</b>: readiness <b>{communication_payload["readiness"]}</b>,
                next 24h risk <b>{communication_payload["next_24h_level"]}</b> i next 7d peak
                <b>{communication_payload["next_7d_peak_level"]}</b> ({communication_payload["next_7d_peak_score"]:.1f}).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Public advisory (HR)")
        st.code(comm_package["public_advisory_hr"], language="text")

        st.markdown("#### Tourist advisory (EN)")
        st.code(comm_package["tourist_advisory_en"], language="text")

        st.markdown("#### Media brief")
        st.code(comm_package["media_brief"], language="text")

        st.markdown("#### Operator SMS / dispatch summary")
        st.code(comm_package["operator_sms"], language="text")

        st.markdown("#### Social post (HR)")
        st.code(comm_package["social_post_hr"], language="text")

        st.markdown("#### Social post (EN)")
        st.code(comm_package["social_post_en"], language="text")

        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown("### Download communication package")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                "⬇ Download HR public advisory (.txt)",
                data=comm_package["public_advisory_hr"].encode("utf-8"),
                file_name=f"heatsafe_hr_public_advisory_{selected_city}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_public_hr_{selected_city}",
            )

        with d2:
            st.download_button(
                "⬇ Download EN tourist advisory (.txt)",
                data=comm_package["tourist_advisory_en"].encode("utf-8"),
                file_name=f"heatsafe_hr_tourist_advisory_{selected_city}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_tourist_en_{selected_city}",
            )

        with d3:
            full_comm_package_text = (
                "HEATSAFE HR — COMMUNICATION PACKAGE\n\n"
                f"{comm_package['public_advisory_hr']}\n\n"
                f"{comm_package['tourist_advisory_en']}\n\n"
                f"{comm_package['media_brief']}\n\n"
                f"{comm_package['operator_sms']}\n\n"
                f"{comm_package['social_post_hr']}\n\n"
                f"{comm_package['social_post_en']}"
            )

            st.download_button(
                "⬇ Download full communication package (.txt)",
                data=full_comm_package_text.encode("utf-8"),
                file_name=f"heatsafe_hr_communication_package_{selected_city}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_comm_package_{selected_city}",
            )
        st.markdown("</div>", unsafe_allow_html=True)