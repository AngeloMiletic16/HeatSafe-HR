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

RISK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_daily_with_risk.csv"

RISK_ORDER = ["Nizak", "Umjeren", "Visok", "Vrlo visok"]
RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}


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


st.title("📊 City Overview")

df = load_risk_data()
cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

col_sidebar_1, col_sidebar_2 = st.columns([1, 1])
with col_sidebar_1:
    selected_city = st.selectbox("Odaberi grad", cities, index=default_index)
with col_sidebar_2:
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
    Detaljni pregled za **{selected_city}**.  
    Ova stranica prikazuje povijesni trend toplinskog rizika, temperaturne obrasce
    i mjesečni pregled ponašanja grada kroz odabrani period.
    """
)

metric_cols = st.columns(5)
metric_cols[0].metric("Zadnja razina rizika", str(full_current["risk_level"]))
metric_cols[1].metric("Zadnji Heat Risk Score", format_metric(full_current["heat_risk_score"]))
metric_cols[2].metric("Prosječni score u periodu", format_metric(filtered_df["heat_risk_score"].mean()))
metric_cols[3].metric("Maks. apparent temp", format_metric(filtered_df["apparent_temp_max"].max(), " °C"))
metric_cols[4].metric(
    "Visoki + vrlo visoki dani",
    int(((filtered_df["risk_level"] == "Visok") | (filtered_df["risk_level"] == "Vrlo visok")).sum()),
)

st.divider()

left, right = st.columns([1.5, 1])

with left:
    st.markdown("### Trend Heat Risk Score")
    fig_risk = px.line(
        filtered_df,
        x="date",
        y="heat_risk_score",
        title=f"Heat Risk Score kroz vrijeme — {selected_city}",
        markers=False,
    )
    fig_risk.update_layout(
        xaxis_title="Datum",
        yaxis_title="Heat Risk Score",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with right:
    st.markdown("### Raspodjela razina rizika")
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
        title=f"Distribucija rizika — {selected_city}",
    )
    fig_pie.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

st.markdown("### Temperaturni signali")
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
    margin=dict(l=20, r=20, t=50, b=20),
)
st.plotly_chart(fig_temp, use_container_width=True)

st.divider()

monthly_df = build_monthly_summary(filtered_df)

col_month_1, col_month_2 = st.columns(2)

with col_month_1:
    st.markdown("### Mjesečni prosječni Heat Risk Score")
    fig_month_risk = px.bar(
        monthly_df,
        x="month",
        y="avg_heat_risk_score",
        title=f"Mjesečni prosjeci rizika — {selected_city}",
    )
    fig_month_risk.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="Prosječni score",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_month_risk, use_container_width=True)

with col_month_2:
    st.markdown("### Broj visokorizičnih dana po mjesecu")
    fig_month_high = px.bar(
        monthly_df,
        x="month",
        y="high_risk_days",
        title=f"Visoki i vrlo visoki rizik — {selected_city}",
    )
    fig_month_high.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="Broj dana",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_month_high, use_container_width=True)

st.divider()

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

st.markdown("### Sažetak za odabrani grad")
st.write(
    f"""
    U odabranom periodu za **{selected_city}** prosječni Heat Risk Score iznosi
    **{filtered_df['heat_risk_score'].mean():.1f}**, a maksimalna apparent temperatura
    dosegnula je **{filtered_df['apparent_temp_max'].max():.1f} °C**.
    Sustav je u tom razdoblju zabilježio
    **{((filtered_df['risk_level'] == 'Visok') | (filtered_df['risk_level'] == 'Vrlo visok')).sum()}**
    visokorizičnih dana.
    """
)