from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY

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


def get_latest_city_snapshot(df: pd.DataFrame, city: str) -> pd.Series:
    city_df = df[df["city"] == city].sort_values("date")
    return city_df.iloc[-1]


def get_latest_all_cities(df: pd.DataFrame) -> pd.DataFrame:
    latest_per_city = (
        df.sort_values(["city", "date"])
        .groupby("city", as_index=False)
        .tail(1)
        .sort_values(["heat_risk_score", "apparent_temp_max"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return latest_per_city


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


def format_metric(value: float | int, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.1f}{suffix}"


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


# ---------- Load ----------
inject_custom_css()

df = load_risk_data()
metrics_v1 = load_json_if_exists(METRICS_V1_PATH)
metrics_v2 = load_json_if_exists(METRICS_V2_PATH)

cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

if "selected_city" not in st.session_state:
    st.session_state.selected_city = DEFAULT_CITY if DEFAULT_CITY in cities else cities[0]

latest_all_cities = get_latest_all_cities(df)
latest_available_date = pd.to_datetime(latest_all_cities["date"]).max()

# ---------- Sidebar ----------
st.sidebar.title("HeatSafe HR")
selected_city = st.sidebar.selectbox(
    "Odaberi grad",
    cities,
    index=cities.index(st.session_state.selected_city) if st.session_state.selected_city in cities else default_index,
)
st.session_state.selected_city = selected_city

city_snapshot = get_latest_city_snapshot(df, selected_city)

metrics_v1_best = metrics_v1.get(metrics_v1.get("best_model", ""), {})
metrics_v2_best = metrics_v2.get(metrics_v2.get("best_model", ""), {})

st.sidebar.markdown("### Status")
st.sidebar.markdown(f"**Grad:** {selected_city}")
st.sidebar.markdown(f"**Risk level:** {city_snapshot['risk_level']}")
st.sidebar.markdown(f"**Readiness:** {readiness_from_level(str(city_snapshot['risk_level']))}")

st.sidebar.markdown("### Brza navigacija")
st.sidebar.page_link("Home.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/1_Overview.py", label="Overview", icon="📊")
st.sidebar.page_link("pages/2_History.py", label="History", icon="🕘")
st.sidebar.page_link("pages/3_Insights.py", label="Insights", icon="🧠")
st.sidebar.page_link("pages/4_Forecast.py", label="Forecast", icon="🔮")
st.sidebar.page_link("pages/5_Action_Center.py", label="Action Center", icon="🚨")
st.sidebar.page_link("pages/7_Methodology_Research.py", label="Methodology / Research", icon="🧪")

# ---------- Hero ----------
st.markdown(
    f"""
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

st.divider()

# ---------- Selected city summary ----------
st.markdown('<div class="section-title">Selected city command view</div>', unsafe_allow_html=True)

left, middle, right = st.columns([1.15, 1.15, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {selected_city}")
    render_status_pill(str(city_snapshot["risk_level"]))
    st.markdown("")
    st.markdown(f"**Readiness status:** {readiness_from_level(str(city_snapshot['risk_level']))}")
    st.markdown(f"**Heat Risk Score:** {city_snapshot['heat_risk_score']:.1f}")
    st.markdown(f"**Temp max:** {city_snapshot['temp_max']:.1f} °C")
    st.markdown(f"**Apparent temp max:** {city_snapshot['apparent_temp_max']:.1f} °C")
    st.markdown(f"**Humidity mean:** {city_snapshot['humidity_mean']:.1f} %")
    st.markdown(f"**Wind speed mean:** {city_snapshot['wind_speed_mean']:.1f} m/s")
    st.markdown(f"**Date:** {pd.to_datetime(city_snapshot['date']).strftime('%d.%m.%Y.')}")
    st.markdown("</div>", unsafe_allow_html=True)

with middle:
    gauge_fig = build_gauge(city_snapshot["heat_risk_score"], "Current Heat Risk Score")
    st.plotly_chart(gauge_fig, use_container_width=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick interpretation")
    st.write(
        f"""
        Za grad **{selected_city}** zadnji zapis pokazuje razinu rizika
        **{city_snapshot['risk_level']}** uz score **{city_snapshot['heat_risk_score']:.1f}**.
        Sustav trenutno procjenjuje readiness status **{readiness_from_level(str(city_snapshot['risk_level']))}**.
        """
    )
    if str(city_snapshot["risk_level"]) == "Nizak":
        st.success("Rutinsko praćenje uvjeta.")
    elif str(city_snapshot["risk_level"]) == "Umjeren":
        st.warning("Pojačano praćenje i priprema komunikacije.")
    elif str(city_snapshot["risk_level"]) == "Visok":
        st.warning("Povećana pripravnost i operativni fokus.")
    else:
        st.error("Kritična pripravnost i pojačane mjere.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------- Top cities ----------
st.markdown('<div class="section-title">Top city risk snapshot</div>', unsafe_allow_html=True)

top_cols = st.columns(min(3, len(latest_all_cities)))
for i, (_, row) in enumerate(latest_all_cities.head(3).iterrows()):
    with top_cols[i]:
        color = risk_color(str(row["risk_level"]))
        st.markdown(
            f"""
            <div class="top-city-card">
                <div class="mini-title">City rank #{i+1}</div>
                <div class="big-city">{row['city']}</div>
                <div style="margin:0.35rem 0 0.55rem 0;">
                    <span class="status-pill" style="background:{color};">{row['risk_level']}</span>
                </div>
                <div class="small-muted">Heat Risk Score: <b>{row['heat_risk_score']:.1f}</b></div>
                <div class="small-muted">Apparent temp max: <b>{row['apparent_temp_max']:.1f} °C</b></div>
                <div class="small-muted">Temp max: <b>{row['temp_max']:.1f} °C</b></div>
                <div class="small-muted">Date: <b>{pd.to_datetime(row['date']).strftime('%d.%m.%Y.')}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

display_cols = [
    "city",
    "date",
    "risk_level",
    "heat_risk_score",
    "temp_max",
    "apparent_temp_max",
    "humidity_mean",
    "wind_speed_mean",
]
overview_df = latest_all_cities[display_cols].copy()
overview_df["date"] = pd.to_datetime(overview_df["date"]).dt.strftime("%d.%m.%Y.")
st.dataframe(overview_df, use_container_width=True, hide_index=True)

st.divider()

# ---------- Product modules ----------
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

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        <div class="card">
            <h4>Production model (v1)</h4>
            <p>
            Operativni model za platformu, optimiziran za praktičnu upotrebu i decision-support
            kontekst unutar sustava.
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
        <b>Current product status:</b> HeatSafe HR već sada kombinira data pipeline,
        AI/ML modeliranje, forecast simulation i operativni alerting u jedinstven alat
        za toplinski rizik. Sljedeći koraci su dodatni polish, export/reporting i finalni
        competition narrative.
    </div>
    """,
    unsafe_allow_html=True,
)