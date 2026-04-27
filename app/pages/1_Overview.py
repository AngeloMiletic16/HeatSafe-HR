from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.sidebar import render_app_sidebar

RISK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_daily_with_risk.csv"

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

st.set_page_config(page_title="City Overview", page_icon="📊", layout="wide")

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
        padding: 1.35rem 1.5rem 1.2rem 1.5rem;
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
        line-height: 1.6;
        opacity: 0.95;
        max-width: 1050px;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        min-height: 110px;
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
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.15rem;
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
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.35rem 0 0.85rem 0;
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

    .summary-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        line-height: 1.8;
        color: #334155;
    }

    .stDataFrame, .stPlotlyChart {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_risk_data() -> pd.DataFrame:
    if not RISK_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RISK_DATA_PATH}. Run risk engine first."
        )

    df = pd.read_csv(RISK_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_by_window(df: pd.DataFrame, window_label: str) -> pd.DataFrame:
    if window_label == "Zadnjih 30 dana":
        cutoff = df["date"].max() - pd.Timedelta(days=30)
        return df[df["date"] >= cutoff].copy()
    if window_label == "Zadnjih 90 dana":
        cutoff = df["date"].max() - pd.Timedelta(days=90)
        return df[df["date"] >= cutoff].copy()
    if window_label == "Zadnjih 365 dana":
        cutoff = df["date"].max() - pd.Timedelta(days=365)
        return df[df["date"] >= cutoff].copy()
    return df.copy()


def format_metric(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}{suffix}"


def latest_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("date").iloc[-1]


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.to_period("M").astype(str)

    monthly = (
        out.groupby("month", as_index=False)
        .agg(
            avg_heat_risk_score=("heat_risk_score", "mean"),
            avg_temp_max=("temp_max", "mean"),
            max_apparent_temp=("apparent_temp_max", "max"),
            high_risk_days=("risk_level", lambda s: ((s == "Visok") | (s == "Vrlo visok")).sum()),
        )
    )

    monthly["avg_heat_risk_score"] = monthly["avg_heat_risk_score"].round(2)
    monthly["avg_temp_max"] = monthly["avg_temp_max"].round(2)
    monthly["max_apparent_temp"] = monthly["max_apparent_temp"].round(2)
    return monthly


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


def panel(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">{title}</div>
            <div class="panel-text">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def readiness_from_risk(level: str) -> str:
    return READINESS_MAP.get(level, "Monitoring")


df = load_risk_data()
cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

selected_city = st.session_state.get(
    "selected_city",
    DEFAULT_CITY if DEFAULT_CITY in cities else cities[0],
)
if selected_city not in cities:
    selected_city = cities[0]

city_df_for_sidebar = df[df["city"] == selected_city].sort_values("date").copy()
sidebar_row = latest_row(city_df_for_sidebar)

render_app_sidebar(
    selected_city=selected_city,
    risk_level=str(sidebar_row["risk_level"]),
    readiness_status=readiness_from_risk(str(sidebar_row["risk_level"])),
)

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">📊 City Overview</div>
        <div class="page-hero-subtitle">
            Povijesni i operativni pregled toplinskog ponašanja grada kroz vrijeme.
            Ova stranica sažima trendove Heat Risk Score-a, temperaturne signale,
            raspodjelu razina rizika i mjesečne obrasce kako bi korisnik brzo razumio
            kako se odabrani grad ponaša kroz sezonu i gdje nastaju kritični periodi.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

control_left, control_right = st.columns([1, 1])
with control_left:
    selected_city = st.selectbox("Odaberi grad", cities, index=cities.index(selected_city))
    st.session_state.selected_city = selected_city
with control_right:
    selected_window = st.selectbox(
        "Period pregleda",
        ["Zadnjih 30 dana", "Zadnjih 90 dana", "Zadnjih 365 dana", "Sve"],
        index=2,
    )

city_df = df[df["city"] == selected_city].sort_values("date").copy()
filtered_df = filter_by_window(city_df, selected_window)

if filtered_df.empty:
    st.warning("Nema podataka za odabrani period.")
    st.stop()

current = latest_row(filtered_df)
full_current = latest_row(city_df)

st.markdown(
    f"""
    <div class="note-box">
        <b>Overview context:</b> Ova stranica služi kao analitički pregled za grad <b>{selected_city}</b>.
        Prikazuje kako se toplinski rizik razvijao u odabranom periodu, koliko je bilo visokorizičnih dana
        i koje temperaturne varijable najviše oblikuju ukupni signal u povijesnim podacima.
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
with metric_cols[0]:
    metric_card("Latest risk level", str(full_current["risk_level"]), "Latest available record")
with metric_cols[1]:
    metric_card("Latest Heat Risk Score", format_metric(full_current["heat_risk_score"]), "Current city status")
with metric_cols[2]:
    metric_card("Average score in window", format_metric(filtered_df["heat_risk_score"].mean()), selected_window)
with metric_cols[3]:
    metric_card("Max apparent temp", format_metric(filtered_df["apparent_temp_max"].max(), " °C"), "Selected period")
with metric_cols[4]:
    metric_card(
        "High-risk days",
        str(int(((filtered_df["risk_level"] == "Visok") | (filtered_df["risk_level"] == "Vrlo visok")).sum())),
        "Visok + Vrlo visok",
    )

st.markdown('<div class="section-title">Trend and distribution</div>', unsafe_allow_html=True)

left, right = st.columns([1.55, 1])

with left:
    fig_risk = px.line(
        filtered_df,
        x="date",
        y="heat_risk_score",
        title=f"Heat Risk Score kroz vrijeme — {selected_city}",
    )
    fig_risk.update_layout(
        xaxis_title="Datum",
        yaxis_title="Heat Risk Score",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_risk.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_risk, use_container_width=True)

with right:
    risk_counts = (
        filtered_df["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["risk_level", "count"]

    fig_pie = px.pie(
        risk_counts,
        names="risk_level",
        values="count",
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        title=f"Distribucija razina rizika — {selected_city}",
    )
    fig_pie.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

info1, info2 = st.columns(2)
with info1:
    panel(
        "Kako čitati trend",
        """
        Heat Risk Score linija pomaže vidjeti je li grad u odabranom razdoblju imao
        stabilan toplinski profil ili izraženije epizode s rastom opterećenja.
        Takvi uzlazni segmenti često su važni za operativnu pripremu.
        """,
    )
with info2:
    panel(
        "Zašto je distribucija važna",
        """
        Sama prosječna vrijednost ne govori sve. Raspodjela po razinama rizika pokazuje
        je li grad većinom bio u niskom režimu ili je imao značajniji broj dana s povišenim
        toplinskim stresom.
        """,
    )

st.divider()

st.markdown('<div class="section-title">Temperature signals</div>', unsafe_allow_html=True)

temp_plot_df = filtered_df[["date", "temp_max", "temp_min", "apparent_temp_max"]].copy()
temp_plot_df = temp_plot_df.melt(id_vars="date", var_name="metric", value_name="value")

fig_temp = px.line(
    temp_plot_df,
    x="date",
    y="value",
    color="metric",
    title=f"Temperaturni trendovi — {selected_city}",
)
fig_temp.update_layout(
    xaxis_title="Datum",
    yaxis_title="°C",
    margin=dict(l=20, r=20, t=55, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig_temp.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
st.plotly_chart(fig_temp, use_container_width=True)

st.divider()

monthly_df = build_monthly_summary(filtered_df)

st.markdown('<div class="section-title">Monthly pattern analysis</div>', unsafe_allow_html=True)

col_month_1, col_month_2 = st.columns(2)

with col_month_1:
    fig_month_risk = px.bar(
        monthly_df,
        x="month",
        y="avg_heat_risk_score",
        title=f"Mjesečni prosječni Heat Risk Score — {selected_city}",
    )
    fig_month_risk.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="Prosječni score",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_month_risk.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_month_risk, use_container_width=True)

with col_month_2:
    fig_month_high = px.bar(
        monthly_df,
        x="month",
        y="high_risk_days",
        title=f"Broj visokorizičnih dana — {selected_city}",
    )
    fig_month_high.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="Broj dana",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_month_high.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_month_high, use_container_width=True)

st.divider()

st.markdown('<div class="section-title">Operational tables</div>', unsafe_allow_html=True)

col_table_1, col_table_2 = st.columns(2)

with col_table_1:
    st.markdown("### Najtopliji dani po apparent temperaturi")
    hottest_days = (
        filtered_df.sort_values("apparent_temp_max", ascending=False)[
            ["date", "risk_level", "heat_risk_score", "temp_max", "apparent_temp_max", "humidity_mean"]
        ]
        .head(10)
        .copy()
    )
    hottest_days["date"] = pd.to_datetime(hottest_days["date"]).dt.strftime("%d.%m.%Y.")
    st.dataframe(hottest_days, use_container_width=True, hide_index=True)

with col_table_2:
    st.markdown("### Zadnjih 14 dana")
    latest_14 = (
        city_df.sort_values("date", ascending=False)[
            ["date", "risk_level", "heat_risk_score", "temp_max", "apparent_temp_max", "wind_speed_mean"]
        ]
        .head(14)
        .copy()
    )
    latest_14["date"] = pd.to_datetime(latest_14["date"]).dt.strftime("%d.%m.%Y.")
    st.dataframe(latest_14, use_container_width=True, hide_index=True)

st.divider()

st.markdown('<div class="section-title">Interpretation summary</div>', unsafe_allow_html=True)

high_risk_days = int(((filtered_df["risk_level"] == "Visok") | (filtered_df["risk_level"] == "Vrlo visok")).sum())
avg_score = filtered_df["heat_risk_score"].mean()
max_apparent = filtered_df["apparent_temp_max"].max()

st.markdown(
    f"""
    <div class="summary-card">
        U odabranom periodu za <b>{selected_city}</b> prosječni Heat Risk Score iznosi
        <b>{avg_score:.1f}</b>, dok je maksimalna apparent temperatura dosegnula
        <b>{max_apparent:.1f} °C</b>. U istom razdoblju sustav je zabilježio
        <b>{high_risk_days}</b> visokorizičnih dana.
        <br><br>
        Ovakav pregled pomaže razumjeti je li grad u pravilu stabilan ili ima izraženije
        toplinske epizode koje se ponavljaju kroz sezonu. To je važna podloga za kasnije
        operativne module kao što su forecast, action center, alerting i historical replay.
    </div>
    """,
    unsafe_allow_html=True,
)