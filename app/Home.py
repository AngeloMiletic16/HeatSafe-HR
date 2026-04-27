from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary
from src.escalation_engine import predict_escalation_from_features
from src.forecast_engine import make_ml_forecast
from src.sidebar import render_app_sidebar
from src.update_live_data import (
    get_data_freshness_info,
    load_refresh_audit_log,
    load_refresh_status,
    refresh_operational_data,
)
from src.vulnerability_engine import (
    build_vulnerability_recommendations,
    get_city_vulnerability_snapshot,
    identify_vulnerability_drivers,
)

st.set_page_config(
    page_title="HeatSafe HR",
    page_icon="🌡️",
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

RISK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_daily_with_risk.csv"
METRICS_V1_PATH = PROJECT_ROOT / "data" / "models" / "model_metrics.json"
METRICS_V2_PATH = PROJECT_ROOT / "data" / "models" / "model_metrics_strict.json"
METRICS_V3_PATH = PROJECT_ROOT / "data" / "models" / "model_metrics_escalation.json"

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
ESCALATION_COLOR_MAP = {
    "Stable": "#64748b",
    "Watch": "#E6A700",
    "Likely escalation": "#C0392B",
}

DASHBOARD_CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

IMPACT_COUNTERS = [
    {
        "label": "Cities in platform",
        "value": "7",
        "sub": "Operational city coverage in Croatia",
    },
    {
        "label": "Mapped resources",
        "value": "58+",
        "sub": "Cooling / support resource points",
    },
    {
        "label": "Critical points",
        "value": "35+",
        "sub": "Hospitals, tourism hubs, elderly care, dense areas",
    },
    {
        "label": "Human framing",
        "value": "60,000+",
        "sub": "Estimated heat-related deaths across Europe in 2023",
    },
]


@st.cache_data
def load_risk_data() -> pd.DataFrame:
    if not RISK_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RISK_DATA_PATH}. Run preprocessing and risk engine first."
        )

    df = pd.read_csv(RISK_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
            border-radius: 22px;
            padding: 1.65rem 1.75rem 1.45rem 1.75rem;
            color: white;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        }

        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.03rem;
            line-height: 1.65;
            opacity: 0.95;
            margin-bottom: 1rem;
            max-width: 1100px;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.45rem;
        }

        .chip {
            display: inline-block;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: white;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .card {
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            height: 100%;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            min-height: 118px;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #475569;
            font-weight: 600;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .metric-sub {
            font-size: 0.88rem;
            color: #64748b;
            line-height: 1.5;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0.35rem 0 0.85rem 0;
            color: #0f172a;
        }

        .soft-note {
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 0.95rem 1rem;
            border-radius: 12px;
            color: #0f172a;
            margin-top: 0.6rem;
            margin-bottom: 0.6rem;
            line-height: 1.65;
        }

        .warning-note {
            background: #fff7ed;
            border-left: 5px solid #f97316;
            padding: 0.95rem 1rem;
            border-radius: 12px;
            color: #7c2d12;
            margin-top: 0.6rem;
            margin-bottom: 0.6rem;
            line-height: 1.65;
        }

        .status-pill {
            display: inline-block;
            padding: 0.42rem 0.82rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.94rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .top-city-card {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            background: #ffffff;
            height: 100%;
        }

        .mini-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .big-city {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .small-muted {
            font-size: 0.88rem;
            color: #64748b;
            line-height: 1.55;
        }

        .story-card {
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            background: #ffffff;
            height: 100%;
        }

        .story-number {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: #0f172a;
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            margin-bottom: 0.7rem;
        }

        .module-card {
            border-radius: 18px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            background: #ffffff;
            overflow: hidden;
            margin-bottom: 0.55rem;
            height: 100%;
        }

        .module-head {
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .module-body {
            padding: 0.95rem 1rem 0.45rem 1rem;
        }

        .closing-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        }

        .stDataFrame, .stPlotlyChart {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_color(level: str) -> str:
    return RISK_COLOR_MAP.get(level, "#666666")


def render_status_pill(level: str) -> None:
    color = risk_color(level)
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{level}</span>',
        unsafe_allow_html=True,
    )


def escalation_badge(text: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{text}</span>',
        unsafe_allow_html=True,
    )


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


def build_gauge(score: float, title: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            title={"text": title},
            number={"font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f172a"},
                "steps": [
                    {"range": [0, 24], "color": "#d1fae5"},
                    {"range": [25, 49], "color": "#fef3c7"},
                    {"range": [50, 74], "color": "#fed7aa"},
                    {"range": [75, 100], "color": "#fecaca"},
                ],
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def build_model_summary_table(metrics_v1: dict, metrics_v2: dict, metrics_v3: dict) -> pd.DataFrame:
    rows = []

    if metrics_v1:
        best_model_name = metrics_v1.get("best_model", "N/A")
        model_metrics = metrics_v1.get(best_model_name, {})
        rows.append(
            {
                "Model version": "Production model (v1)",
                "Best model": best_model_name,
                "Accuracy": model_metrics.get("accuracy"),
                "Macro F1": model_metrics.get("macro_f1"),
                "Weighted F1": model_metrics.get("weighted_f1"),
                "F1 positive": None,
                "ROC AUC": None,
            }
        )

    if metrics_v2:
        best_model_name = metrics_v2.get("best_model", "N/A")
        model_metrics = metrics_v2.get(best_model_name, {})
        rows.append(
            {
                "Model version": "Strict model (v2)",
                "Best model": best_model_name,
                "Accuracy": model_metrics.get("accuracy"),
                "Macro F1": model_metrics.get("macro_f1"),
                "Weighted F1": model_metrics.get("weighted_f1"),
                "F1 positive": None,
                "ROC AUC": None,
            }
        )

    if metrics_v3:
        best_model_name = metrics_v3.get("best_model", "N/A")
        model_metrics = metrics_v3.get("models", {}).get(best_model_name, {})
        rows.append(
            {
                "Model version": "Escalation model (v3)",
                "Best model": best_model_name,
                "Accuracy": model_metrics.get("accuracy"),
                "Macro F1": None,
                "Weighted F1": None,
                "F1 positive": model_metrics.get("f1_positive"),
                "ROC AUC": model_metrics.get("roc_auc"),
            }
        )

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    for col in ["Accuracy", "Macro F1", "Weighted F1", "F1 positive", "ROC AUC"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: round(float(x), 3) if pd.notna(x) else None
            )
    return summary_df


def build_live_escalation_snapshot(city: str, forecast_df: pd.DataFrame) -> dict:
    if forecast_df.empty:
        return {
            "escalation_probability_72h": None,
            "escalation_flag_72h": None,
            "escalation_label_72h": None,
            "operator_message": "Escalation signal nije dostupan.",
        }

    snapshot_df = forecast_df.sort_values("date").head(1).copy()

    if "city" not in snapshot_df.columns:
        snapshot_df["city"] = city

    pred_df = predict_escalation_from_features(snapshot_df)
    row = pred_df.iloc[0]

    probability = float(row["escalation_probability_72h"])
    label = str(row["escalation_label_72h"])
    flag = int(row["escalation_flag_72h"])

    if label == "Stable":
        operator_message = (
            f"72h escalation probability is {probability:.2f}. "
            "Signal je nizak i grad trenutno ne pokazuje kratkoročnu eskalaciju."
        )
    elif label == "Watch":
        operator_message = (
            f"72h escalation probability is {probability:.2f}. "
            "Potrebno je pojačano praćenje jer postoji srednji signal pogoršanja unutar 72 sata."
        )
    else:
        operator_message = (
            f"72h escalation probability is {probability:.2f}. "
            "Likely escalation detected; preporučuje se rana priprema i proaktivna komunikacija."
        )

    return {
        "escalation_probability_72h": probability,
        "escalation_flag_72h": flag,
        "escalation_label_72h": label,
        "operator_message": operator_message,
    }


@st.cache_data(ttl=1800)
def build_command_dashboard_snapshot(cities: list[str]) -> pd.DataFrame:
    rows = []

    for city in cities:
        forecast_df = make_ml_forecast(city)

        if "city" not in forecast_df.columns:
            forecast_df["city"] = city

        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        summary = build_city_readiness_summary(city, forecast_df)
        escalation = build_live_escalation_snapshot(city, forecast_df)

        first_row = forecast_df.sort_values("date").iloc[0]

        rows.append(
            {
                "city": city,
                "date": pd.to_datetime(first_row["date"]),
                "risk_level": first_row["heuristic_risk_level"],
                "heat_risk_score": float(first_row["heuristic_risk_score"]),
                "temp_max": float(first_row["temp_max"]),
                "apparent_temp_max": float(first_row["apparent_temp_max"]),
                "humidity_mean": float(first_row["humidity_mean"]),
                "wind_speed_mean": float(first_row["wind_speed_mean"]) if "wind_speed_mean" in first_row else None,
                "readiness_status": summary["readiness_status"],
                "next_7d_peak_level": summary["next_7d_peak_level"],
                "next_7d_peak_score": float(summary["next_7d_peak_score"]),
                "high_risk_days": int(summary["high_risk_days"]),
                "escalation_probability_72h": escalation["escalation_probability_72h"],
                "escalation_label_72h": escalation["escalation_label_72h"],
                "escalation_flag_72h": escalation["escalation_flag_72h"],
                "escalation_operator_message": escalation["operator_message"],
            }
        )

    return pd.DataFrame(rows)


inject_custom_css()

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

if "demo_city" not in st.session_state:
    st.session_state.demo_city = "Split"

topbar_left, topbar_right = st.columns([1.25, 1])

with topbar_left:
    freshness_info = get_data_freshness_info()
    st.markdown(
        f'<span class="status-pill" style="background:{freshness_info["badge_color"]};">{freshness_info["badge_label"]}</span>',
        unsafe_allow_html=True,
    )

with topbar_right:
    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("🔄 Refresh data", use_container_width=True):
            refresh_result = refresh_operational_data()
            st.cache_data.clear()
            if refresh_result.get("status") == "success":
                st.success("Operational data refreshed.")
            else:
                st.error(refresh_result.get("message", "Refresh failed."))
    with btn2:
        if st.button("🎬 Demo mode", use_container_width=True):
            st.session_state.demo_mode = True
            st.session_state.demo_city = "Split"
            st.session_state.selected_city = "Split"
            st.rerun()

refresh_status = load_refresh_status()
refresh_audit_log = load_refresh_audit_log(limit=5)

df = load_risk_data()
metrics_v1 = load_json_if_exists(METRICS_V1_PATH)
metrics_v2 = load_json_if_exists(METRICS_V2_PATH)
metrics_v3 = load_json_if_exists(METRICS_V3_PATH)

cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

if "selected_city" not in st.session_state:
    st.session_state.selected_city = DEFAULT_CITY if DEFAULT_CITY in cities else cities[0]

dashboard_snapshot_df = build_command_dashboard_snapshot(DASHBOARD_CITIES)

latest_available_date = max(
    pd.to_datetime(df["date"]).max(),
    pd.to_datetime(dashboard_snapshot_df["date"]).max(),
)

ranked_cities_df = dashboard_snapshot_df.sort_values(
    ["escalation_probability_72h", "next_7d_peak_score", "heat_risk_score"],
    ascending=[False, False, False],
).reset_index(drop=True)

if st.session_state.demo_mode:
    selected_city = st.session_state.demo_city
else:
    selected_city = st.sidebar.selectbox(
        "Odaberi grad",
        cities,
        index=cities.index(st.session_state.selected_city) if st.session_state.selected_city in cities else default_index,
    )

st.session_state.selected_city = selected_city

selected_city_snapshot = ranked_cities_df[ranked_cities_df["city"] == selected_city].iloc[0]
vulnerability_snapshot = get_city_vulnerability_snapshot(selected_city)
vulnerability_drivers = identify_vulnerability_drivers(vulnerability_snapshot)
vulnerability_recommendations = build_vulnerability_recommendations(vulnerability_snapshot)

metrics_v1_best = metrics_v1.get(metrics_v1.get("best_model", ""), {})
metrics_v2_best = metrics_v2.get(metrics_v2.get("best_model", ""), {})
metrics_v3_best_name = metrics_v3.get("best_model", "N/A")
metrics_v3_best = metrics_v3.get("models", {}).get(metrics_v3_best_name, {})

render_app_sidebar(
    selected_city=selected_city,
    risk_level=selected_city_snapshot["risk_level"],
    readiness_status=selected_city_snapshot["readiness_status"],
    escalation_label=selected_city_snapshot["escalation_label_72h"],
    escalation_probability=selected_city_snapshot["escalation_probability_72h"],
)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🌡️ HeatSafe HR</div>
        <div class="hero-subtitle">
            AI platforma za rano upozoravanje na toplinske valove i procjenu toplinskog stresa
            u hrvatskim gradovima. Sustav spaja meteorološke podatke, risk engine, strojno učenje,
            vulnerabilnost, resurse, explainable AI, dispatch routing i operativne preporuke
            za stvarni odgovor na toplinske rizike.
        </div>
        <div class="chip-row">
            <span class="chip">Smart City</span>
            <span class="chip">AI / ML</span>
            <span class="chip">Climate Resilience</span>
            <span class="chip">Public Safety</span>
            <span class="chip">Tourism Readiness</span>
            <span class="chip">Decision Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="soft-note">
        <b>Data freshness:</b> {freshness_info["message"]}<br>
        <b>Last refresh status:</b> {refresh_status.get("status", "unknown")}<br>
        <b>Cities updated:</b> {refresh_status.get("cities_updated", 0)}
    </div>
    """,
    unsafe_allow_html=True,
)

if freshness_info["freshness_state"] in ["stale", "error"]:
    st.markdown(
        """
        <div class="warning-note">
            <b>Operational warning:</b> Podaci nisu svježi. Prije donošenja odluka preporučuje se pokrenuti refresh
            i provjeriti audit trail.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Platform story</div>', unsafe_allow_html=True)

story1, story2, story3 = st.columns(3)
with story1:
    st.markdown(
        """
        <div class="story-card">
            <div class="story-number">1</div>
            <div class="big-city">Detect</div>
            <div class="small-muted">
                HeatSafe HR rano detektira rast toplinskog rizika kroz multiclass ML modele,
                72h escalation predictor i scenario simulation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with story2:
    st.markdown(
        """
        <div class="story-card">
            <div class="story-number">2</div>
            <div class="big-city">Decide</div>
            <div class="small-muted">
                Platforma kombinira readiness, vulnerability, XAI i dispatch routing kako bi
                operateri znali gdje je rizik najveći, tko je najizloženiji i koji resurs prvo aktivirati.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with story3:
    st.markdown(
        """
        <div class="story-card">
            <div class="story-number">3</div>
            <div class="big-city">Communicate</div>
            <div class="small-muted">
                Sustav generira public advisory, alert pakete i official-style PDF daily briefing
                spreman za službe, turizam i javnu komunikaciju.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")
demo_col1, demo_col2 = st.columns([1.2, 1])
with demo_col1:
    if st.session_state.demo_mode:
        st.markdown(
            f"""
            <div class="soft-note">
                <b>Demo mode active:</b> fokusiran je grad <b>{st.session_state.demo_city}</b> kao primjer
                city-level heat emergency workflowa. Preporučeni flow je:
                <b>Command Dashboard → Action Center → Alert Center → PDF Daily Briefing</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
with demo_col2:
    if st.session_state.demo_mode:
        if st.button("Exit Demo Mode", use_container_width=True):
            st.session_state.demo_mode = False
            st.rerun()

st.divider()

st.markdown('<div class="section-title">Human impact framing</div>', unsafe_allow_html=True)

impact_cols = st.columns(4)
for idx, item in enumerate(IMPACT_COUNTERS):
    with impact_cols[idx]:
        render_metric_card(item["label"], item["value"], item["sub"])

st.markdown(
    """
    <div class="soft-note">
        <b>Why this matters:</b> toplinski valovi više nisu samo meteorološki problem. Oni utječu na zdravlje,
        turizam, javne službe, starije osobe, djecu i radnike na otvorenom. HeatSafe HR je zamišljen kao alat
        koji gradovima omogućuje raniju pripremu, ciljanu komunikaciju i bolju raspodjelu resursa prije nego
        toplinski rizik preraste u stvarni operativni problem.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card(
        "Zadnji dostupni datum",
        latest_available_date.strftime("%d.%m.%Y."),
        "Data pipeline status",
    )
with k2:
    render_metric_card(
        "Production model Macro F1",
        str(metrics_v1_best.get("macro_f1", "N/A")),
        str(metrics_v1.get("best_model", "N/A")),
    )
with k3:
    render_metric_card(
        "Strict model Macro F1",
        str(metrics_v2_best.get("macro_f1", "N/A")),
        str(metrics_v2.get("best_model", "N/A")),
    )
with k4:
    render_metric_card(
        "Escalation model F1+",
        str(metrics_v3_best.get("f1_positive", "N/A")),
        metrics_v3_best_name,
    )

highest_escalation_row = ranked_cities_df.sort_values(
    ["escalation_probability_72h", "next_7d_peak_score"],
    ascending=[False, False],
).iloc[0]

st.markdown(
    f"""
    <div class="soft-note">
        <b>V3 early-warning signal:</b> trenutno najveću 72h vjerojatnost eskalacije ima grad
        <b>{highest_escalation_row['city']}</b> uz probability
        <b>{highest_escalation_row['escalation_probability_72h']:.2f}</b>
        i signal <b>{highest_escalation_row['escalation_label_72h']}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown('<div class="section-title">Selected city command view</div>', unsafe_allow_html=True)

risk_color_hex = risk_color(str(selected_city_snapshot["risk_level"]))
esc_color_hex = ESCALATION_COLOR_MAP.get(selected_city_snapshot["escalation_label_72h"], "#64748b")
date_str = pd.to_datetime(selected_city_snapshot["date"]).strftime("%d.%m.%Y.")

left, middle, right = st.columns([1.05, 1.1, 1])

with left:
    st.markdown(
        f"""
        <div class="card" style="padding:1.2rem 1.3rem;">
            <div style="font-size:1.5rem;font-weight:800;color:#0f172a;margin-bottom:0.7rem;">{selected_city}</div>
            <div style="display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap;">
                <span style="background:{risk_color_hex};color:white;padding:0.3rem 0.9rem;border-radius:999px;font-size:0.85rem;font-weight:700;">{selected_city_snapshot["risk_level"]}</span>
                <span style="background:{esc_color_hex};color:white;padding:0.3rem 0.9rem;border-radius:999px;font-size:0.85rem;font-weight:700;">{selected_city_snapshot["escalation_label_72h"]}</span>
            </div>
            <div style="border-top:1px solid #f1f5f9;padding-top:0.9rem;display:flex;flex-direction:column;gap:0.55rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Readiness status</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["readiness_status"]}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">72h escalation prob.</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["escalation_probability_72h"]:.2f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Next 7d peak</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["next_7d_peak_level"]} ({selected_city_snapshot["next_7d_peak_score"]:.1f})</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Vulnerability band</span>
                    <span style="color:#0f172a;font-weight:700;">{vulnerability_snapshot["vulnerability_band"]}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Heat Risk Score today</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["heat_risk_score"]:.1f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Temp max</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["temp_max"]:.1f} °C</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Apparent temp max</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["apparent_temp_max"]:.1f} °C</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Humidity mean</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["humidity_mean"]:.1f} %</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span style="color:#64748b;font-weight:500;">Wind speed mean</span>
                    <span style="color:#0f172a;font-weight:700;">{selected_city_snapshot["wind_speed_mean"]:.1f} m/s</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;border-top:1px solid #f1f5f9;padding-top:0.55rem;margin-top:0.2rem;">
                    <span style="color:#64748b;font-weight:500;">Date</span>
                    <span style="color:#0f172a;font-weight:700;">{date_str}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with middle:
    gauge_fig = build_gauge(selected_city_snapshot["heat_risk_score"], "Current Heat Risk Score")
    st.plotly_chart(gauge_fig, use_container_width=True)

with right:
    st.markdown(
        f"""
        <div class="card" style="padding:1.2rem 1.3rem;">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#64748b;font-weight:700;margin-bottom:0.5rem;">Quick interpretation</div>
            <div style="font-size:0.95rem;color:#0f172a;line-height:1.72;margin-bottom:1rem;">
                Za grad <b>{selected_city_snapshot["city"]}</b> trenutni signal pokazuje razinu rizika
                <b>{selected_city_snapshot["risk_level"]}</b> i readiness status
                <b>{selected_city_snapshot["readiness_status"]}</b>.
                Operativno važniji indikator je da model procjenjuje
                <b>72h escalation probability = {selected_city_snapshot["escalation_probability_72h"]:.2f}</b>,
                dok je <b>next 7d peak = {selected_city_snapshot["next_7d_peak_level"]} ({selected_city_snapshot["next_7d_peak_score"]:.1f})</b>.
            </div>
            <div style="border-top:1px solid #f1f5f9;padding-top:0.9rem;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#64748b;font-weight:700;margin-bottom:0.5rem;">V3 Early-warning signal</div>
                <div style="font-size:0.95rem;color:#0f172a;line-height:1.7;margin-bottom:0.8rem;">
                    72h escalation probability: <b>{selected_city_snapshot["escalation_probability_72h"]:.2f}</b><br>
                    Signal: <b>{selected_city_snapshot["escalation_label_72h"]}</b><br>
                    Vulnerability context: <b>{vulnerability_snapshot["vulnerability_band"]}</b>
                </div>
                <div style="background:#f8fafc;border-left:3px solid {esc_color_hex};border-radius:0 10px 10px 0;padding:0.7rem 0.9rem;font-size:0.88rem;color:#1e293b;line-height:1.6;">
                    {selected_city_snapshot["escalation_operator_message"]}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">72h escalation early-warning</div>', unsafe_allow_html=True)

esc1, esc2, esc3 = st.columns(3)
with esc1:
    render_metric_card(
        "72h escalation probability",
        f"{selected_city_snapshot['escalation_probability_72h']:.2f}",
        "V3 early-warning model",
    )
with esc2:
    render_metric_card(
        "Escalation signal",
        selected_city_snapshot["escalation_label_72h"],
        selected_city_snapshot["readiness_status"],
    )
with esc3:
    render_metric_card(
        "Next 7d peak",
        selected_city_snapshot["next_7d_peak_level"],
        f"{selected_city_snapshot['next_7d_peak_score']:.1f}",
    )

st.markdown('<div class="section-title">Vulnerability preview</div>', unsafe_allow_html=True)

vul_band_color = {
    "Low vulnerability": "#2E8B57",
    "Moderate vulnerability": "#E6A700",
    "High vulnerability": "#E67E22",
    "Very high vulnerability": "#C0392B",
}.get(vulnerability_snapshot["vulnerability_band"], "#64748b")

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        color: white;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 2.5rem;
        flex-wrap: wrap;
    ">
        <div>
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;opacity:0.55;margin-bottom:0.3rem;">Vulnerability Index</div>
            <div style="font-size:2.6rem;font-weight:800;line-height:1;">{vulnerability_snapshot['vulnerability_index']:.1f}</div>
            <div style="font-size:0.82rem;opacity:0.5;margin-top:0.25rem;">City-level profile</div>
        </div>
        <div style="width:1px;background:rgba(255,255,255,0.1);height:64px;"></div>
        <div>
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;opacity:0.55;margin-bottom:0.5rem;">Vulnerability Band</div>
            <span style="background:{vul_band_color};padding:0.35rem 1rem;border-radius:999px;font-size:0.95rem;font-weight:700;">
                {vulnerability_snapshot['vulnerability_band']}
            </span>
        </div>
        <div style="width:1px;background:rgba(255,255,255,0.1);height:64px;"></div>
        <div>
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;opacity:0.55;margin-bottom:0.3rem;">Active Drivers</div>
            <div style="font-size:2.6rem;font-weight:800;line-height:1;">{len(vulnerability_drivers)}</div>
            <div style="font-size:0.82rem;opacity:0.5;margin-top:0.25rem;">Operational prioritization input</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

vd1, vd2 = st.columns(2)
with vd1:
    drivers_html = "".join(
        f"""
        <div style="display:flex;align-items:flex-start;gap:0.6rem;padding:0.55rem 0;border-bottom:1px solid #f1f5f9;">
            <span style="color:#f97316;font-size:1rem;margin-top:0.05rem;">⚠</span>
            <span style="font-size:0.93rem;color:#1e293b;line-height:1.4;">{item}</span>
        </div>
        """
        for item in vulnerability_drivers
    )
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:0.6rem;padding-bottom:0.5rem;border-bottom:2px solid #f1f5f9;">
                Main vulnerability drivers
            </div>
            {drivers_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with vd2:
    recs_html = "".join(
        f"""
        <div style="display:flex;align-items:flex-start;gap:0.6rem;padding:0.55rem 0;border-bottom:1px solid #f1f5f9;">
            <span style="color:#2E8B57;font-size:1rem;margin-top:0.05rem;">✓</span>
            <span style="font-size:0.93rem;color:#1e293b;line-height:1.4;">{item}</span>
        </div>
        """
        for item in vulnerability_recommendations
    )
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:0.6rem;padding-bottom:0.5rem;border-bottom:2px solid #f1f5f9;">
                Vulnerability-sensitive recommendations
            </div>
            {recs_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown('<div class="section-title">Refresh audit trail</div>', unsafe_allow_html=True)

if not refresh_audit_log:
    st.info("Audit trail je trenutno prazan. Pokreni refresh data da se generira zapis.")
else:
    audit_df = pd.DataFrame(refresh_audit_log)
    if "timestamp_utc" in audit_df.columns:
        audit_df["timestamp_utc"] = pd.to_datetime(audit_df["timestamp_utc"]).dt.strftime("%d.%m.%Y. %H:%M")
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

st.divider()

st.markdown('<div class="section-title">Top city risk snapshot</div>', unsafe_allow_html=True)

top_cols = st.columns(min(3, len(ranked_cities_df)))
for i, (_, row) in enumerate(ranked_cities_df.head(3).iterrows()):
    with top_cols[i]:
        risk_badge_color = RISK_COLOR_MAP.get(str(row["risk_level"]), "#64748b")
        escalation_badge_color = ESCALATION_COLOR_MAP.get(str(row["escalation_label_72h"]), "#64748b")

        st.markdown(
            f"""
            <div class="top-city-card">
                <div class="mini-title">City rank #{i+1}</div>
                <div class="big-city">{row['city']}</div>
                <div style="margin:0.35rem 0 0.55rem 0;">
                    <span class="status-pill" style="background:{risk_badge_color};">{row['risk_level']}</span>
                    <span class="status-pill" style="background:{escalation_badge_color};">{row['escalation_label_72h']}</span>
                </div>
                <div class="small-muted">72h escalation prob.: <b>{row['escalation_probability_72h']:.2f}</b></div>
                <div class="small-muted">Next 7d peak: <b>{row['next_7d_peak_level']} ({row['next_7d_peak_score']:.1f})</b></div>
                <div class="small-muted">Readiness: <b>{row['readiness_status']}</b></div>
                <div class="small-muted">Heat Risk Score today: <b>{row['heat_risk_score']:.1f}</b></div>
                <div class="small-muted">Date: <b>{pd.to_datetime(row['date']).strftime('%d.%m.%Y.')}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

command_table_df = ranked_cities_df[
    [
        "city",
        "date",
        "readiness_status",
        "escalation_label_72h",
        "escalation_probability_72h",
        "next_7d_peak_level",
        "next_7d_peak_score",
        "high_risk_days",
        "risk_level",
        "heat_risk_score",
        "temp_max",
        "apparent_temp_max",
        "humidity_mean",
    ]
].copy()
command_table_df["date"] = pd.to_datetime(command_table_df["date"]).dt.strftime("%d.%m.%Y.")
st.dataframe(command_table_df, use_container_width=True, hide_index=True)

st.divider()

st.markdown('<div class="section-title">Platform modules</div>', unsafe_allow_html=True)

modules = [
    {
        "category": "Operational Monitoring",
        "icon": "📊",
        "title": "Overview & History",
        "description": "Povijesni trendovi, gradski rizik, sezonski obrasci i najkritičniji toplinski periodi.",
        "color": "#3b82f6",
        "links": [
            ("pages/1_Overview.py", "📊", "Overview"),
            ("pages/2_History.py", "🕘", "History"),
        ],
    },
    {
        "category": "AI / ML Intelligence",
        "icon": "🧠",
        "title": "Insights & Forecast",
        "description": "Usporedba modela, confusion matrix, feature importance, threshold tuning i XAI analiza.",
        "color": "#8b5cf6",
        "links": [
            ("pages/3_Insights.py", "🧠", "Insights"),
            ("pages/4_Forecast.py", "🔮", "Forecast"),
        ],
    },
    {
        "category": "Decision Support",
        "icon": "🚨",
        "title": "Action Center & Alerts",
        "description": "Forecast, readiness, operator actions, public advisory, alerting i export-ready daily briefing.",
        "color": "#ef4444",
        "links": [
            ("pages/5_Action_Center.py", "🚨", "Action Center"),
            ("pages/10_Alert_Center.py", "📢", "Alert Center"),
        ],
    },
    {
        "category": "Command & Control",
        "icon": "🖥️",
        "title": "Command Dashboard",
        "description": "Operator cockpit, model consensus, city ranking i dispatch summary za sve gradove.",
        "color": "#0f172a",
        "links": [
            ("pages/6_Command_Dashboard.py", "🖥️", "Command Dashboard"),
        ],
    },
    {
        "category": "Public & Resources",
        "icon": "🌍",
        "title": "Public Advisory & Map",
        "description": "Javne preporuke za građane, turiste i ranjive grupe. Mapa resursa i cooling centara.",
        "color": "#2E8B57",
        "links": [
            ("pages/8_Public_Advisory.py", "📣", "Public Advisory"),
            ("pages/9_Resources_Map.py", "🗺️", "Resources Map"),
        ],
    },
    {
        "category": "Simulation & Research",
        "icon": "⚡",
        "title": "Stress Test & Methodology",
        "description": "Ekstremni scenariji, what-if simulacija, stress report i metodološka dokumentacija.",
        "color": "#E67E22",
        "links": [
            ("pages/12_Stress_Test.py", "⚡", "Stress Test"),
            ("pages/7_Methodology_Research.py", "📚", "Methodology"),
        ],
    },
]

row1 = st.columns(3)
row2 = st.columns(3)
all_cols = row1 + row2

for col, mod in zip(all_cols, modules):
    with col:
        st.markdown(
            f"""
            <div class="module-card">
                <div class="module-head" style="background:{mod['color']};">
                    <span style="font-size:1.2rem;">{mod['icon']}</span>
                    <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;color:rgba(255,255,255,0.85);font-weight:700;">
                        {mod['category']}
                    </span>
                </div>
                <div class="module-body">
                    <div style="font-size:1.1rem;font-weight:800;color:#0f172a;margin-bottom:0.4rem;">{mod['title']}</div>
                    <div style="font-size:0.88rem;color:#64748b;line-height:1.55;">{mod['description']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for page_path, icon, label in mod["links"]:
            st.page_link(page_path, label=label, icon=icon)
        st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-title">Model architecture summary</div>', unsafe_allow_html=True)

model_summary_df = build_model_summary_table(metrics_v1, metrics_v2, metrics_v3)
if not model_summary_df.empty:
    st.dataframe(model_summary_df, use_container_width=True, hide_index=True)
else:
    st.warning("Model metrics još nisu dostupne.")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
            <h4>Production model (v1)</h4>
            <p style="color:#475569; line-height:1.7;">
            Operativni multiclass model optimiziran za praktičnu procjenu razine toplinskog rizika
            i dnevni decision-support rad platforme.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <h4>Strict model (v2)</h4>
            <p style="color:#475569; line-height:1.7;">
            Metodološki stroža validacijska verzija važna za istraživačku ozbiljnost projekta
            i provjeru da sustav ne ovisi samo o shortcut signalima.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="card">
            <h4>Escalation model (v3)</h4>
            <p style="color:#475569; line-height:1.7;">
            72h early-warning model koji procjenjuje vjerojatnost ulaska grada u
            ozbiljniji toplinski rizik unutar sljedeća tri dana.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown('<div class="section-title">Who benefits from HeatSafe HR?</div>', unsafe_allow_html=True)

benefit1, benefit2, benefit3 = st.columns(3)

with benefit1:
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-bottom:0.55rem;">🏙️ Gradovi i komunalne službe</h3>
            <p style="color:#475569; line-height:1.7; margin-bottom:0.8rem;">
                HeatSafe HR pomaže gradovima da prije toplinskog udara ne reagiraju naslijepo,
                nego da unaprijed vide gdje raste rizik, kada treba podići readiness i kako
                prioritetizirati ranjive skupine, gradske resurse i operativne korake.
            </p>
            <ul style="color:#334155; line-height:1.75; padding-left:1.1rem;">
                <li>rano upozorenje na rast toplinskog rizika</li>
                <li>aktivacija preventivnih mjera prije vršnog opterećenja</li>
                <li>gradska koordinacija kroz readiness, routing i briefing</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with benefit2:
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-bottom:0.55rem;">🚑 Javne i hitne službe</h3>
            <p style="color:#475569; line-height:1.7; margin-bottom:0.8rem;">
                Za civilnu zaštitu, hitne službe i zdravstveni sustav platforma djeluje kao
                decision-support sloj: ne prikazuje samo prognozu, nego daje signal kada treba
                pojačati pažnju, koje skupine su najizloženije i kako pripremiti operativni odgovor.
            </p>
            <ul style="color:#334155; line-height:1.75; padding-left:1.1rem;">
                <li>pojačana pripravnost prije toplinskih epizoda</li>
                <li>praćenje rizičnih dana, gradova i vulnerabilnih skupina</li>
                <li>operativni brief i konkretni action items za teren</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with benefit3:
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-bottom:0.55rem;">🏖️ Turizam, događaji i javnost</h3>
            <p style="color:#475569; line-height:1.7; margin-bottom:0.8rem;">
                U zemlji u kojoj ljetni mjeseci znače velik pritisak na turizam i javne prostore,
                HeatSafe HR omogućuje sigurnije planiranje aktivnosti, bolju komunikaciju prema
                gostima i građanima te jednostavnije prilagodbe rasporeda kod visokog toplinskog rizika.
            </p>
            <ul style="color:#334155; line-height:1.75; padding-left:1.1rem;">
                <li>event risk check za aktivnosti na otvorenom</li>
                <li>prilagodba rasporeda i komunikacije prema gostima</li>
                <li>jasne javne preporuke na hrvatskom i engleskom jeziku</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

