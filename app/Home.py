from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable
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
from src.update_live_data import load_refresh_status, refresh_operational_data
from src.resource_recommender import recommend_resources

st.set_page_config(
    page_title="HeatSafe HR",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_daily_with_risk.csv"
METRICS_V1_PATH = PROJECT_ROOT / "data" / "models" / "model_metrics.json"
METRICS_V2_PATH = PROJECT_ROOT / "data" / "models" / "model_metrics_strict.json"

RISK_ORDER = ["Nizak", "Umjeren", "Visok", "Vrlo visok"]
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
            padding: 1.6rem 1.7rem 1.4rem 1.7rem;
            color: white;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.6;
            opacity: 0.95;
            margin-bottom: 1rem;
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
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            height: 100%;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
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
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0.3rem 0 0.8rem 0;
            color: #0f172a;
        }

        .soft-note {
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            color: #0f172a;
            margin-top: 0.6rem;
            margin-bottom: 0.6rem;
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
            margin-bottom: 0.2rem;
        }

        .small-muted {
            font-size: 0.88rem;
            color: #64748b;
        }

        .cta-card {
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
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


def readiness_from_level(level: str) -> str:
    return READINESS_MAP.get(level, "Monitoring")


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
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def build_model_summary_table(metrics_v1: dict, metrics_v2: dict) -> pd.DataFrame:
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
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


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


# ---------- Load ----------
inject_custom_css()

df = load_risk_data()
metrics_v1 = load_json_if_exists(METRICS_V1_PATH)
metrics_v2 = load_json_if_exists(METRICS_V2_PATH)

cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

if "selected_city" not in st.session_state:
    st.session_state.selected_city = DEFAULT_CITY if DEFAULT_CITY in cities else cities[0]

dashboard_snapshot_df = build_command_dashboard_snapshot(DASHBOARD_CITIES)

latest_available_date = max(
    pd.to_datetime(df["date"]).max(),
    pd.to_datetime(dashboard_snapshot_df["date"]).max(),
)

last_refresh_status = load_refresh_status()

ranked_cities_df = dashboard_snapshot_df.sort_values(
    ["escalation_probability_72h", "next_7d_peak_score", "heat_risk_score"],
    ascending=[False, False, False],
).reset_index(drop=True)

# ---------- Sidebar ----------
st.sidebar.title("HeatSafe HR")
selected_city = st.sidebar.selectbox(
    "Odaberi grad",
    cities,
    index=cities.index(st.session_state.selected_city) if st.session_state.selected_city in cities else default_index,
)
st.session_state.selected_city = selected_city

st.sidebar.markdown("### Live data")

if st.sidebar.button("🔄 Osvježi podatke", use_container_width=True):
    with st.spinner("Osvježavam podatke za sve gradove..."):
        refresh_status = refresh_operational_data(
            rebuild_escalation_dataset=True,
            retrain_models=False,
            fail_fast=True,
        )

    if refresh_status["success"]:
        st.cache_data.clear()
        st.success("Podaci su uspješno osvježeni.")
        st.rerun()
    else:
        st.error("Došlo je do greške tijekom osvježavanja podataka.")
        with st.expander("Detalji refresha"):
            st.code(json.dumps(refresh_status, indent=2, ensure_ascii=False))

if last_refresh_status:
    finished_at = last_refresh_status.get("finished_at")
    success_flag = last_refresh_status.get("success")

    if finished_at:
        st.sidebar.caption(
            f"Zadnji refresh: {pd.to_datetime(finished_at).strftime('%d.%m.%Y. %H:%M')}"
        )
    st.sidebar.caption(f"Status: {'OK' if success_flag else 'Error'}")
else:
    st.sidebar.caption("Zadnji refresh: još nije pokrenut")

selected_city_snapshot = ranked_cities_df[ranked_cities_df["city"] == selected_city].iloc[0]
recommended_resources_df = recommend_resources(
    city=selected_city,
    escalation_label=selected_city_snapshot["escalation_label_72h"],
    top_n=3,
)

metrics_v1_best = metrics_v1.get(metrics_v1.get("best_model", ""), {})
metrics_v2_best = metrics_v2.get(metrics_v2.get("best_model", ""), {})

st.sidebar.markdown("### Status")
st.sidebar.markdown(f"**Grad:** {selected_city}")
st.sidebar.markdown(f"**Risk level:** {selected_city_snapshot['risk_level']}")
st.sidebar.markdown(f"**Readiness:** {selected_city_snapshot['readiness_status']}")
st.sidebar.markdown(
    f"**72h escalation:** {selected_city_snapshot['escalation_label_72h']} "
    f"({selected_city_snapshot['escalation_probability_72h']:.2f})"
)

st.sidebar.markdown("### Brza navigacija")
st.sidebar.page_link("Home.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/1_Overview.py", label="Overview", icon="📊")
st.sidebar.page_link("pages/2_History.py", label="History", icon="🕘")
st.sidebar.page_link("pages/3_Insights.py", label="Insights", icon="🧠")
st.sidebar.page_link("pages/4_Forecast.py", label="Forecast", icon="🔮")
st.sidebar.page_link("pages/5_Action_Center.py", label="Action Center", icon="🚨")
st.sidebar.page_link("pages/6_Command_Dashboard.py", label="Command Dashboard", icon="🧭")
st.sidebar.page_link("pages/7_Methodology_Research.py", label="Methodology / Research", icon="🧪")
st.sidebar.page_link("pages/8_Public_Advisory.py", label="Public Advisory", icon="📣")
st.sidebar.page_link("pages/9_Resources_Map.py", label="Resources Map", icon="🧊")
st.sidebar.page_link("pages/10_Alert_Center.py", label="Alert Center", icon="🚨")
st.sidebar.page_link("pages/11_Historical_Replay.py", label="Historical Replay", icon="⏪")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🌡️ HeatSafe HR</div>
        <div class="hero-subtitle">
            AI platforma za rano upozoravanje na toplinske valove i procjenu toplinskog stresa
            u hrvatskim gradovima. Sustav spaja meteorološke podatke, risk engine, strojno učenje,
            scenarije i operativne preporuke za gradove, javne službe i turizam.
        </div>
        <div class="chip-row">
            <span class="chip">Smart City</span>
            <span class="chip">AI / ML</span>
            <span class="chip">Climate Resilience</span>
            <span class="chip">Public Safety</span>
            <span class="chip">Tourism Readiness</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- KPI row ----------
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
        "Gradova u sustavu",
        str(len(cities)),
        "Croatia v1 coverage",
    )

st.markdown(
    """
    <div class="soft-note">
        <b>Competition framing:</b> HeatSafe HR nije samo weather dashboard,
        nego AI/ML decision-support sustav koji gradovima i službama pomaže da
        nekoliko dana unaprijed prepoznaju toplinski rizik i pripreme odgovor.
    </div>
    """,
    unsafe_allow_html=True,
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

if last_refresh_status:
    finished_at = last_refresh_status.get("finished_at")
    success_flag = last_refresh_status.get("success")

    if finished_at:
        st.markdown(
            f"""
            <div class="soft-note">
                <b>Live refresh status:</b> zadnje osvježavanje podataka završilo je
                <b>{pd.to_datetime(finished_at).strftime('%d.%m.%Y. u %H:%M')}</b>
                uz status <b>{'OK' if success_flag else 'Error'}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ---------- Selected city summary ----------
st.markdown('<div class="section-title">Selected city command view</div>', unsafe_allow_html=True)

left, middle, right = st.columns([1.05, 1.1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {selected_city}")
    render_status_pill(str(selected_city_snapshot["risk_level"]))
    escalation_badge(
        selected_city_snapshot["escalation_label_72h"],
        ESCALATION_COLOR_MAP.get(selected_city_snapshot["escalation_label_72h"], "#64748b"),
    )
    st.markdown("")
    st.markdown(f"**Readiness status:** {selected_city_snapshot['readiness_status']}")
    st.markdown(f"**Heat Risk Score:** {selected_city_snapshot['heat_risk_score']:.1f}")
    st.markdown(f"**72h escalation probability:** {selected_city_snapshot['escalation_probability_72h']:.2f}")
    st.markdown(f"**Temp max:** {selected_city_snapshot['temp_max']:.1f} °C")
    st.markdown(f"**Apparent temp max:** {selected_city_snapshot['apparent_temp_max']:.1f} °C")
    st.markdown(f"**Humidity mean:** {selected_city_snapshot['humidity_mean']:.1f} %")
    if pd.notna(selected_city_snapshot["wind_speed_mean"]):
        st.markdown(f"**Wind speed mean:** {selected_city_snapshot['wind_speed_mean']:.1f} m/s")
    st.markdown(f"**Date:** {pd.to_datetime(selected_city_snapshot['date']).strftime('%d.%m.%Y.')}")
    st.markdown("</div>", unsafe_allow_html=True)

with middle:
    gauge_fig = build_gauge(selected_city_snapshot["heat_risk_score"], "Current Heat Risk Score")
    st.plotly_chart(gauge_fig, use_container_width=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick interpretation")
    quick_text = (
        f"Za grad **{selected_city_snapshot['city']}** trenutni forecast signal pokazuje "
        f"razinu rizika **{selected_city_snapshot['risk_level']}** uz score "
        f"**{selected_city_snapshot['heat_risk_score']:.1f}**. "
        f"Sustav procjenjuje readiness status **{selected_city_snapshot['readiness_status']}**.\n\n"
        f"V3 early-warning model daje **72h escalation probability = "
        f"{selected_city_snapshot['escalation_probability_72h']:.2f}** "
        f"i signal **{selected_city_snapshot['escalation_label_72h']}**."
    )
    st.write(quick_text)

    if str(selected_city_snapshot["risk_level"]) == "Nizak":
        st.success("Rutinsko praćenje uvjeta.")
    elif str(selected_city_snapshot["risk_level"]) == "Umjeren":
        st.warning("Pojačano praćenje i priprema komunikacije.")
    elif str(selected_city_snapshot["risk_level"]) == "Visok":
        st.warning("Povećana pripravnost i operativni fokus.")
    else:
        st.error("Kritična pripravnost i pojačane mjere.")

    st.info(selected_city_snapshot["escalation_operator_message"])
    st.markdown("</div>", unsafe_allow_html=True)

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

st.divider()

st.markdown('<div class="section-title">Recommended resources for current escalation signal</div>', unsafe_allow_html=True)

if recommended_resources_df.empty:
    st.info("Nema preporučenih resource točaka za ovaj grad.")
else:
    rcols = st.columns(min(3, len(recommended_resources_df)))

    for i, (_, row) in enumerate(recommended_resources_df.iterrows()):
        with rcols[i]:
            st.markdown(
                f"""
                <div class="card">
                    <div class="mini-title">{row.get('resource_type', 'Resource')}</div>
                    <div class="big-city">{row.get('resource_name', 'Unknown')}</div>
                    <div class="small-muted"><b>Adresa:</b> {row.get('address', 'N/A')}</div>
                    <div class="small-muted"><b>Radno vrijeme:</b> {row.get('hours_weekday', 'N/A')}</div>
                    <div class="small-muted"><b>Verified:</b> {row.get('verified_status', 'N/A')}</div>
                    <div class="small-muted"><b>Water:</b> {row.get('water_available', 'N/A')}</div>
                    <div class="small-muted"><b>Indoor cooling:</b> {row.get('indoor_cooling', 'N/A')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------- Top cities ----------
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
                <div class="small-muted">Heat Risk Score: <b>{row['heat_risk_score']:.1f}</b></div>
                <div class="small-muted">72h escalation prob.: <b>{row['escalation_probability_72h']:.2f}</b></div>
                <div class="small-muted">Next 7d peak: <b>{row['next_7d_peak_level']} ({row['next_7d_peak_score']:.1f})</b></div>
                <div class="small-muted">Readiness: <b>{row['readiness_status']}</b></div>
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
        "risk_level",
        "heat_risk_score",
        "next_7d_peak_level",
        "next_7d_peak_score",
        "high_risk_days",
        "escalation_probability_72h",
        "escalation_label_72h",
        "readiness_status",
        "temp_max",
        "apparent_temp_max",
        "humidity_mean",
    ]
].copy()
command_table_df["date"] = pd.to_datetime(command_table_df["date"]).dt.strftime("%d.%m.%Y.")
st.dataframe(command_table_df, use_container_width=True, hide_index=True)

st.divider()

# ---------- Platform modules ----------
st.markdown('<div class="section-title">Platform modules</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        """
        <div class="cta-card">
            <div class="mini-title">Operational monitoring</div>
            <div class="big-city">Overview & History</div>
            <div class="small-muted">
                Povijesni trendovi, gradski rizik, sezonski obrasci i najkritičniji toplinski periodi.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Overview.py", label="Open Overview", icon="📊")
    st.page_link("pages/2_History.py", label="Open History", icon="🕘")

with m2:
    st.markdown(
        """
        <div class="cta-card">
            <div class="mini-title">AI / ML intelligence</div>
            <div class="big-city">Insights</div>
            <div class="small-muted">
                Usporedba modela, confusion matrix, feature importance i analiza pogrešaka.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Insights.py", label="Open Insights", icon="🧠")

with m3:
    st.markdown(
        """
        <div class="cta-card">
            <div class="mini-title">Decision support</div>
            <div class="big-city">Forecast & Action Center</div>
            <div class="small-muted">
                ML forecast, scenario simulation, event risk check i executive operational brief.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Forecast.py", label="Open Forecast", icon="🔮")
    st.page_link("pages/5_Action_Center.py", label="Open Action Center", icon="🚨")

st.divider()

# ---------- Model summary ----------
st.markdown('<div class="section-title">Model architecture summary</div>', unsafe_allow_html=True)

model_summary_df = build_model_summary_table(metrics_v1, metrics_v2)
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
            <p>
            Operativni multiclass model optimiziran za praktičnu procjenu razine toplinskog rizika.
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
            <p>
            Metodološki stroža validacijska verzija bez oslanjanja na
            <code>heat_risk_score*</code> featuree, važna za istraživačku ozbiljnost projekta.
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
            <p>
            72h early-warning model koji procjenjuje vjerojatnost ulaska grada u
            ozbiljniji toplinski rizik unutar sljedeća tri dana.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------- Value proposition ----------
st.markdown('<div class="section-title">Who benefits from HeatSafe HR?</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="card">
            <h4>🏙️ Gradovi i komunalne službe</h4>
            <ul>
                <li>rano upozorenje na rast toplinskog rizika</li>
                <li>aktivacija preventivnih mjera</li>
                <li>gradska koordinacija i readiness status</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h4>🚑 Javne i hitne službe</h4>
            <ul>
                <li>pojačana pripravnost prije toplinskih epizoda</li>
                <li>praćenje rizičnih dana i grupa</li>
                <li>operativni brief i action items</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h4>🏖️ Turizam i događaji</h4>
            <ul>
                <li>event risk check za aktivnosti na otvorenom</li>
                <li>prilagodba rasporeda i komunikacije prema gostima</li>
                <li>scenario analiza i sigurnosne preporuke</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------- Final note ----------
st.markdown(
    """
    <div class="soft-note">
        <b>Current product status:</b> HeatSafe HR sada kombinira data pipeline,
        multiclass AI/ML procjenu rizika, 72h escalation early-warning model,
        forecast simulation i operativni alerting u jedinstven alat
        za toplinski rizik u hrvatskim gradovima.
    </div>
    """,
    unsafe_allow_html=True,
)