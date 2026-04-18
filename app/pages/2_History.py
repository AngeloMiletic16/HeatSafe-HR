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
        raise FileNotFoundError(f"Missing file: {RISK_DATA_PATH}. Run risk engine first.")

    df = pd.read_csv(RISK_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    return df


def build_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            avg_heat_risk_score=("heat_risk_score", "mean"),
            max_heat_risk_score=("heat_risk_score", "max"),
            high_risk_days=("risk_level", lambda s: ((s == "Visok") | (s == "Vrlo visok")).sum()),
            very_high_days=("risk_level", lambda s: (s == "Vrlo visok").sum()),
            max_apparent_temp=("apparent_temp_max", "max"),
        )
    )
    return yearly


def build_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["month", "month_name"], as_index=False)
        .agg(
            avg_heat_risk_score=("heat_risk_score", "mean"),
            avg_temp_max=("temp_max", "mean"),
            avg_apparent_temp_max=("apparent_temp_max", "mean"),
            high_risk_days=("risk_level", lambda s: ((s == "Visok") | (s == "Vrlo visok")).sum()),
        )
        .sort_values("month")
    )
    return monthly


def build_heatwave_candidates(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["high_risk_flag"] = (out["heat_risk_score"] >= threshold).astype(int)

    groups = []
    current_group = []

    for _, row in out.iterrows():
        if row["high_risk_flag"] == 1:
            current_group.append(row)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    if current_group:
        groups.append(current_group)

    records = []
    for group in groups:
        if len(group) >= 2:
            gdf = pd.DataFrame(group)
            records.append(
                {
                    "start_date": gdf["date"].min(),
                    "end_date": gdf["date"].max(),
                    "duration_days": len(gdf),
                    "max_heat_risk_score": gdf["heat_risk_score"].max(),
                    "max_apparent_temp": gdf["apparent_temp_max"].max(),
                    "avg_temp_max": gdf["temp_max"].mean(),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "duration_days",
                "max_heat_risk_score",
                "max_apparent_temp",
                "avg_temp_max",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["duration_days", "max_heat_risk_score"], ascending=[False, False]
    )


st.title("🕘 History")

df = load_risk_data()
cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

col1, col2 = st.columns([1, 1])
with col1:
    selected_city = st.selectbox("Odaberi grad", cities, index=default_index)
with col2:
    selected_year_mode = st.selectbox(
        "Godine",
        ["Sve godine", "Zadnje 3 godine", "Zadnja 1 godina"],
        index=0,
    )

city_df = df[df["city"] == selected_city].sort_values("date").copy()

if selected_year_mode == "Zadnje 3 godine":
    cutoff = city_df["date"].max() - pd.Timedelta(days=365 * 3)
    city_df = city_df[city_df["date"] >= cutoff].copy()
elif selected_year_mode == "Zadnja 1 godina":
    cutoff = city_df["date"].max() - pd.Timedelta(days=365)
    city_df = city_df[city_df["date"] >= cutoff].copy()

if city_df.empty:
    st.warning("Nema podataka za odabrani period.")
    st.stop()

st.markdown(
    f"""
    Povijesni pregled za **{selected_city}**.  
    Ova stranica prikazuje kako su se toplinski rizici mijenjali kroz godine, mjesece
    i kroz najizraženije toplinske epizode.
    """
)

metric_cols = st.columns(4)
metric_cols[0].metric("Broj dana u datasetu", f"{len(city_df):,}")
metric_cols[1].metric(
    "Prosječni Heat Risk Score",
    f"{city_df['heat_risk_score'].mean():.1f}",
)
metric_cols[2].metric(
    "Maks. apparent temp",
    f"{city_df['apparent_temp_max'].max():.1f} °C",
)
metric_cols[3].metric(
    "Visoki + vrlo visoki dani",
    int(((city_df["risk_level"] == "Visok") | (city_df["risk_level"] == "Vrlo visok")).sum()),
)

st.divider()

yearly_df = build_yearly_summary(city_df)
monthly_df = build_monthly_climatology(city_df)
episodes_df = build_heatwave_candidates(city_df, threshold=50.0)

left, right = st.columns(2)

with left:
    st.markdown("### Godišnji prosječni Heat Risk Score")
    fig_yearly = px.bar(
        yearly_df,
        x="year",
        y="avg_heat_risk_score",
        title=f"Godišnji prosjeci rizika — {selected_city}",
    )
    fig_yearly.update_layout(
        xaxis_title="Godina",
        yaxis_title="Prosječni score",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

with right:
    st.markdown("### Broj visokorizičnih dana po godini")
    fig_high_days = px.bar(
        yearly_df,
        x="year",
        y="high_risk_days",
        title=f"Visoki i vrlo visoki rizik — {selected_city}",
    )
    fig_high_days.update_layout(
        xaxis_title="Godina",
        yaxis_title="Broj dana",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_high_days, use_container_width=True)

st.divider()

col_month_1, col_month_2 = st.columns(2)

with col_month_1:
    st.markdown("### Sezonski obrazac rizika")
    fig_monthly_risk = px.line(
        monthly_df,
        x="month_name",
        y="avg_heat_risk_score",
        markers=True,
        title=f"Mjesečni prosjeci Heat Risk Score — {selected_city}",
    )
    fig_monthly_risk.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="Prosječni score",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_monthly_risk, use_container_width=True)

with col_month_2:
    st.markdown("### Sezonski obrazac apparent temperature")
    fig_monthly_temp = px.line(
        monthly_df,
        x="month_name",
        y="avg_apparent_temp_max",
        markers=True,
        title=f"Mjesečni prosjeci apparent temperature — {selected_city}",
    )
    fig_monthly_temp.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="°C",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_monthly_temp, use_container_width=True)

st.divider()

st.markdown("### Povijesni trend dnevnog Heat Risk Score")
fig_daily = px.line(
    city_df,
    x="date",
    y="heat_risk_score",
    color="risk_level",
    color_discrete_map=RISK_COLOR_MAP,
    title=f"Dnevni Heat Risk Score kroz vrijeme — {selected_city}",
)
fig_daily.update_layout(
    xaxis_title="Datum",
    yaxis_title="Heat Risk Score",
    margin=dict(l=20, r=20, t=50, b=20),
)
st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

col_table_1, col_table_2 = st.columns(2)

with col_table_1:
    st.markdown("### Najizraženije toplinske epizode")
    if episodes_df.empty:
        st.info("Nisu pronađene višednevne epizode s Heat Risk Score ≥ 50.")
    else:
        episodes_show = episodes_df.head(15).copy()
        episodes_show["start_date"] = pd.to_datetime(episodes_show["start_date"]).dt.strftime("%d.%m.%Y.")
        episodes_show["end_date"] = pd.to_datetime(episodes_show["end_date"]).dt.strftime("%d.%m.%Y.")
        episodes_show["avg_temp_max"] = episodes_show["avg_temp_max"].round(1)
        st.dataframe(episodes_show, use_container_width=True, hide_index=True)

with col_table_2:
    st.markdown("### Top 15 dana po najvećem riziku")
    top_days = (
        city_df.sort_values(["heat_risk_score", "apparent_temp_max"], ascending=[False, False])[
            ["date", "risk_level", "heat_risk_score", "temp_max", "apparent_temp_max", "humidity_mean"]
        ]
        .head(15)
        .copy()
    )
    top_days["date"] = pd.to_datetime(top_days["date"]).dt.strftime("%d.%m.%Y.")
    st.dataframe(top_days, use_container_width=True, hide_index=True)

st.divider()

st.markdown("### Sažetak povijesnog ponašanja")
highest_year = yearly_df.sort_values("max_apparent_temp", ascending=False).iloc[0]["year"]
highest_year_score = yearly_df.sort_values("avg_heat_risk_score", ascending=False).iloc[0]["year"]

st.write(
    f"""
    Za grad **{selected_city}** najviša apparent temperatura u promatranom periodu zabilježena je u godini
    **{int(highest_year)}**, dok je najveći prosječni Heat Risk Score ostvaren u godini
    **{int(highest_year_score)}**. Povijesna analiza pomaže u razumijevanju sezonskih obrazaca,
    identifikaciji kritičnih ljetnih razdoblja i planiranju preventivnih mjera.
    """
)