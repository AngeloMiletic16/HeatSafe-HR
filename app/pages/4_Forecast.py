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
from src.decision_engine import build_city_readiness_summary, build_sector_actions
from src.forecast_engine import make_ml_forecast
from src.sidebar import render_app_sidebar

st.set_page_config(
    page_title="Forecast",
    page_icon="🔮",
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

READINESS_COLOR_MAP = {
    "Monitoring": "#2E8B57",
    "Prepared": "#E6A700",
    "Elevated Readiness": "#E67E22",
    "Critical Preparedness": "#C0392B",
}

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

        .page-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
            border-radius: 22px;
            padding: 1.4rem 1.55rem 1.25rem 1.55rem;
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
            font-size: 0.98rem;
            line-height: 1.62;
            opacity: 0.95;
            max-width: 1080px;
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
            margin: 0.35rem 0 0.85rem 0;
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
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.7rem;
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
            margin: 0.7rem 0 1rem 0;
            line-height: 1.65;
        }

        .warning-box {
            background: #fff7ed;
            border-left: 6px solid #ea580c;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            color: #7c2d12;
            margin: 0.7rem 0 1rem 0;
            line-height: 1.65;
        }

        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.78rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.86rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .list-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            height: 100%;
        }

        .list-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.7rem;
        }

        .list-body {
            margin: 0;
            padding-left: 1.1rem;
            color: #334155;
            line-height: 1.72;
            font-size: 0.94rem;
        }

        .summary-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            line-height: 1.78;
            color: #334155;
        }

        .summary-card p {
            margin: 0 0 0.95rem 0;
        }

        .summary-card p:last-child {
            margin-bottom: 0;
        }

        div[data-baseweb="tab-list"] {
            gap: 1rem;
            margin-top: 0.9rem;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
        }

        button[data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            font-weight: 700;
            padding: 0.56rem 0.95rem;
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


def format_value(value: Any, digits: int = 1, suffix: str = "", default: str = "N/A") -> str:
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return default


def risk_color(level: str) -> str:
    return RISK_COLOR_MAP.get(level, "#64748b")


def readiness_color(status: str) -> str:
    return READINESS_COLOR_MAP.get(status, "#64748b")


def to_display_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column]).dt.strftime("%d.%m.%Y.")
    return out


def render_metric_card(label: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{text}</span>',
        unsafe_allow_html=True,
    )


def render_list_card(title: str, items: list[str]) -> None:
    list_html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="list-card">
            <div class="list-title">{title}</div>
            <ul class="list-body">
                {list_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800)
def load_forecast(
    city_name: str,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> pd.DataFrame:
    df = make_ml_forecast(
        city_name,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


inject_custom_css()

default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = CITIES.index(default_city) if default_city in CITIES else 0

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🔮 Forecast / Scenario Intelligence</div>
        <div class="page-hero-subtitle">
            Forecast modul prevodi 7-dnevnu vremensku prognozu u operativni toplinski signal.
            Umjesto klasične prognoze, HeatSafe HR ovdje spaja meteorološke ulaze, projected
            heat-risk logiku i strict ML klasifikaciju kako bi korisnik vidio kako se rizik
            ponaša kroz naredne dane i kako bi se mogao promijeniti u nepovoljnijem scenariju.
        </div>
        <div class="chip-row">
            <span class="chip">7-Day Forecast</span>
            <span class="chip">Strict ML Model</span>
            <span class="chip">Scenario Mode</span>
            <span class="chip">Readiness</span>
            <span class="chip">Peak Detection</span>
            <span class="chip">Decision Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Forecast control panel</div>', unsafe_allow_html=True)

col_top_1, col_top_2 = st.columns([1, 1])

with col_top_1:
    selected_city = st.selectbox("Odaberi grad", CITIES, index=default_index)
    st.session_state.selected_city = selected_city

with col_top_2:
    scenario_enabled = st.toggle("Uključi scenario mode", value=True)

if scenario_enabled:
    st.markdown(
        """
        <div class="note-box">
            <b>Scenario mode:</b> simulira kako bi se signal promijenio ako temperatura poraste,
            vlaga bude veća ili vjetar oslabi. To je korisno za stres-testiranje toplinskog rizika
            prije nego što nepovoljniji uvjeti zaista nastupe.
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        temperature_delta = st.slider("Promjena temperature (°C)", min_value=-2, max_value=12, value=6, step=1)
    with s2:
        humidity_delta = st.slider("Promjena vlage (%)", min_value=-20, max_value=30, value=10, step=1)
    with s3:
        wind_delta = st.slider("Promjena vjetra (m/s)", min_value=-8, max_value=5, value=-3, step=1)
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

try:
    baseline_df = load_forecast(selected_city, 0, 0, 0)
    scenario_df = load_forecast(
        selected_city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
except Exception as exc:
    st.error(f"Forecast nije dostupan: {exc}")
    st.stop()

active_df = scenario_df if scenario_enabled else baseline_df

baseline_summary = build_city_readiness_summary(selected_city, baseline_df)
active_summary = build_city_readiness_summary(selected_city, active_df)
sector_actions = build_sector_actions(active_summary["next_7d_peak_level"])

first_active_row = active_df.sort_values("date").iloc[0]

render_app_sidebar(
    selected_city=selected_city,
    risk_level=str(first_active_row["heuristic_risk_level"]),
    readiness_status=active_summary["readiness_status"],
)

baseline_peak = baseline_df.sort_values(
    ["heuristic_risk_score", "ml_prediction_confidence", "apparent_temp_max"],
    ascending=[False, False, False],
).iloc[0]

scenario_peak = scenario_df.sort_values(
    ["heuristic_risk_score", "ml_prediction_confidence", "apparent_temp_max"],
    ascending=[False, False, False],
).iloc[0]

active_peak = scenario_peak if scenario_enabled else baseline_peak

baseline_high_days = int(
    ((baseline_df["heuristic_risk_level"] == "Visok") | (baseline_df["heuristic_risk_level"] == "Vrlo visok")).sum()
)
scenario_high_days = int(
    ((scenario_df["heuristic_risk_level"] == "Visok") | (scenario_df["heuristic_risk_level"] == "Vrlo visok")).sum()
)

st.markdown(
    f"""
    <div class="note-box">
        <b>Forecast context for {selected_city}:</b> aktivni signal pokazuje
        <b>{active_summary["next_24h_ml_label"]}</b> za sljedećih 24h uz readiness status
        <b>{active_summary["readiness_status"]}</b>. Najvažniji kratkoročni pokazatelj je
        <b>next 7d peak = {active_summary["next_7d_peak_level"]} ({active_summary["next_7d_peak_score"]:.1f})</b>
        na datum <b>{pd.to_datetime(active_summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

if scenario_enabled:
    st.markdown(
        f"""
        <div class="warning-box">
            <b>Scenario inputs:</b> ΔT <b>{temperature_delta:+.1f} °C</b>,
            ΔRH <b>{humidity_delta:+.1f}%</b>,
            ΔWind <b>{wind_delta:+.1f} m/s</b>. Aktivni readiness i interpretacija ispod
            računaju se prema scenario projekciji.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Forecast summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card(
        "Next 24h ML risk",
        str(active_summary["next_24h_ml_label"]),
        "Strict forecast classifier",
    )
with k2:
    render_metric_card(
        "Next 24h confidence",
        format_value(active_summary["next_24h_confidence"], 2),
        "Prediction confidence",
    )
with k3:
    render_metric_card(
        "Peak active score",
        format_value(active_peak["heuristic_risk_score"]),
        str(active_peak["heuristic_risk_level"]),
    )
with k4:
    render_metric_card(
        "High-risk days",
        str(active_summary["high_risk_days"]),
        "Visok + Vrlo visok",
    )

if scenario_enabled:
    st.markdown("### Baseline vs scenario delta")
    s_k1, s_k2, s_k3, s_k4 = st.columns(4)
    with s_k1:
        st.metric(
            "Scenario peak score",
            f"{scenario_peak['heuristic_risk_score']:.1f}",
            delta=f"{scenario_peak['heuristic_risk_score'] - baseline_peak['heuristic_risk_score']:.1f}",
        )
    with s_k2:
        st.metric(
            "Scenario peak apparent temp",
            f"{scenario_peak['apparent_temp_max']:.1f} °C",
            delta=f"{scenario_peak['apparent_temp_max'] - baseline_peak['apparent_temp_max']:.1f} °C",
        )
    with s_k3:
        st.metric(
            "Scenario high-risk days",
            scenario_high_days,
            delta=scenario_high_days - baseline_high_days,
        )
    with s_k4:
        st.metric(
            "Scenario peak ML confidence",
            f"{scenario_peak['ml_prediction_confidence']:.2f}",
        )

st.divider()

st.markdown('<div class="section-title">Projected risk signal</div>', unsafe_allow_html=True)

left, right = st.columns([1.65, 1])

with left:
    compare_df = baseline_df[["date", "heuristic_risk_score"]].copy()
    compare_df["series"] = "Baseline"
    compare_df = compare_df.rename(columns={"heuristic_risk_score": "score"})

    if scenario_enabled:
        scenario_plot_df = scenario_df[["date", "heuristic_risk_score"]].copy()
        scenario_plot_df["series"] = "Scenario"
        scenario_plot_df = scenario_plot_df.rename(columns={"heuristic_risk_score": "score"})
        compare_df = pd.concat([compare_df, scenario_plot_df], ignore_index=True)

    fig_compare = px.line(
        compare_df,
        x="date",
        y="score",
        color="series",
        markers=True,
        title=f"Projected Heat Risk Score — {selected_city}",
    )
    fig_compare.update_layout(
        xaxis_title="Datum",
        yaxis_title="Projected score",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_compare.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_compare, use_container_width=True)

with right:
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">Peak day interpretation</div>
            <div class="panel-text">
                <b>Baseline peak day:</b> {pd.to_datetime(baseline_peak['date']).strftime('%d.%m.%Y.')}<br>
                - heuristic risk: <b>{baseline_peak['heuristic_risk_level']}</b><br>
                - heuristic score: <b>{baseline_peak['heuristic_risk_score']:.1f}</b><br>
                - ML label: <b>{baseline_peak['ml_predicted_label']}</b><br>
                - ML confidence: <b>{baseline_peak['ml_prediction_confidence']:.2f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if scenario_enabled:
        st.markdown(
            f"""
            <div class="soft-panel" style="margin-top:0.85rem;">
                <div class="panel-title">Scenario peak interpretation</div>
                <div class="panel-text">
                    <b>Scenario peak day:</b> {pd.to_datetime(scenario_peak['date']).strftime('%d.%m.%Y.')}<br>
                    - heuristic risk: <b>{scenario_peak['heuristic_risk_level']}</b><br>
                    - heuristic score: <b>{scenario_peak['heuristic_risk_score']:.1f}</b><br>
                    - ML label: <b>{scenario_peak['ml_predicted_label']}</b><br>
                    - ML confidence: <b>{scenario_peak['ml_prediction_confidence']:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title">Sector recommendations</div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
with r1:
    render_list_card("Preporuke za grad", sector_actions["city"])
with r2:
    render_list_card("Preporuke za javne službe", sector_actions["services"])
with r3:
    render_list_card("Preporuke za turizam", sector_actions["tourism"])

st.divider()

st.markdown('<div class="section-title">ML view by day</div>', unsafe_allow_html=True)

proba_cols = [c for c in active_df.columns if c.startswith("proba_class_")]
proba_display = active_df[["date", "ml_predicted_label", "ml_prediction_confidence"] + proba_cols].copy()
proba_display = to_display_date(proba_display)

st.dataframe(proba_display, use_container_width=True, hide_index=True)

st.divider()

st.markdown('<div class="section-title">Weather signals</div>', unsafe_allow_html=True)

weather_plot_df = baseline_df[
    ["date", "temp_max", "temp_min", "apparent_temp_max"]
].copy().melt(id_vars="date", var_name="metric", value_name="value")

fig_weather = px.line(
    weather_plot_df,
    x="date",
    y="value",
    color="metric",
    markers=True,
    title=f"Baseline forecast signals — {selected_city}",
)
fig_weather.update_layout(
    xaxis_title="Datum",
    yaxis_title="°C",
    margin=dict(l=20, r=20, t=55, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig_weather.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
st.plotly_chart(fig_weather, use_container_width=True)

if scenario_enabled:
    st.markdown("### Baseline vs scenario apparent temperature")

    compare_temp_df = baseline_df[["date", "apparent_temp_max"]].copy()
    compare_temp_df["series"] = "Baseline"
    compare_temp_df = compare_temp_df.rename(columns={"apparent_temp_max": "value"})

    scenario_temp_df = scenario_df[["date", "apparent_temp_max"]].copy()
    scenario_temp_df["series"] = "Scenario"
    scenario_temp_df = scenario_temp_df.rename(columns={"apparent_temp_max": "value"})

    compare_temp_df = pd.concat([compare_temp_df, scenario_temp_df], ignore_index=True)

    fig_temp_compare = px.line(
        compare_temp_df,
        x="date",
        y="value",
        color="series",
        markers=True,
        title=f"Baseline vs scenario apparent temperature — {selected_city}",
    )
    fig_temp_compare.update_layout(
        xaxis_title="Datum",
        yaxis_title="°C",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_temp_compare.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_temp_compare, use_container_width=True)

st.divider()

st.markdown('<div class="section-title">Forecast tables</div>', unsafe_allow_html=True)

table_cols = [
    "date",
    "ml_predicted_label",
    "ml_prediction_confidence",
    "heuristic_risk_level",
    "heuristic_risk_score",
    "temp_max",
    "temp_min",
    "apparent_temp_max",
    "humidity_mean",
    "precipitation_sum",
    "wind_speed_max",
]

baseline_table = to_display_date(baseline_df[table_cols].copy())
scenario_table = to_display_date(scenario_df[table_cols].copy())

tab1, tab2 = st.tabs(["Baseline", "Scenario"])

with tab1:
    st.dataframe(baseline_table, use_container_width=True, hide_index=True)

with tab2:
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

st.divider()

st.markdown('<div class="section-title">Interpretation summary</div>', unsafe_allow_html=True)

delta_peak_score = scenario_peak["heuristic_risk_score"] - baseline_peak["heuristic_risk_score"]
delta_apparent = scenario_peak["apparent_temp_max"] - baseline_peak["apparent_temp_max"]
delta_days = scenario_high_days - baseline_high_days

summary_html = (
    f'<div class="summary-card">'
    f'<p>Za grad <b>{selected_city}</b> forecast sloj pokazuje da je aktivni signal za sljedećih 7 dana '
    f'trenutno u režimu <b>{active_summary["readiness_status"]}</b>, uz očekivani peak '
    f'<b>{active_summary["next_7d_peak_level"]}</b> na datum '
    f'<b>{pd.to_datetime(active_summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")}</b>.</p>'
    f'<p>Strict ML model za idući dan daje oznaku <b>{active_summary["next_24h_ml_label"]}</b> '
    f'uz confidence <b>{active_summary["next_24h_confidence"]:.2f}</b>, dok heuristički sloj '
    f'pomaže razumjeti intenzitet i trajanje potencijalne toplinske epizode.</p>'
)

if scenario_enabled:
    summary_html += (
        f'<p>U uključenom scenario modu peak score se mijenja za <b>{delta_peak_score:+.1f}</b>, '
        f'apparent temperatura na peak danu za <b>{delta_apparent:+.1f} °C</b>, '
        f'a broj visokorizičnih dana za <b>{delta_days:+d}</b>. '
        f'To forecast modul pretvara iz obične prognoze u '
        f'<b>AI/ML decision-support alat</b> za procjenu osjetljivosti sustava '
        f'na nepovoljnije ljetne uvjete.</p>'
    )
else:
    summary_html += (
        '<p>U baseline modu korisnik vidi izvorni signal bez dodatnih stresnih promjena, '
        'što daje dobru referentnu točku za kasniju scenario analizu i operativno planiranje.</p>'
    )

summary_html += '</div>'

st.markdown(summary_html, unsafe_allow_html=True)