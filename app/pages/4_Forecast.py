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
from src.forecast_engine import make_ml_forecast

RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}

RISK_ORDER = ["Nizak", "Umjeren", "Visok", "Vrlo visok"]


def recommendation_text(level: str) -> tuple[str, str, str]:
    if level == "Nizak":
        return (
            "Rutinsko praćenje uvjeta i održavanje osnovne pripravnosti.",
            "Standardni operativni režim uz povremene provjere osjetljivih skupina.",
            "Redovita informacija gostima i bez posebnih ograničenja aktivnosti.",
        )
    if level == "Umjeren":
        return (
            "Pojačano javno informiranje i priprema preventivnih mjera.",
            "Praćenje mogućeg povećanja opterećenja i priprema dodatnih timova.",
            "Upozoriti organizatore aktivnosti na otvorenom i preporučiti prilagodbe termina.",
        )
    if level == "Visok":
        return (
            "Aktivirati pojačane gradske obavijesti i pripremiti rashladne punktove.",
            "Pojačati operativnu spremnost i pratiti rizične skupine.",
            "Prilagoditi rasporede aktivnosti, posebno u najtoplijem dijelu dana.",
        )
    return (
        "Aktivirati krizni komunikacijski i preventivni režim za toplinski val.",
        "Povećati pripravnost hitnih i javnih službi te pojačati nadzor rizičnih skupina.",
        "Preporučiti odgodu ili snažnu prilagodbu aktivnosti na otvorenom te informirati goste o mjerama opreza.",
    )


def to_display_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column]).dt.strftime("%d.%m.%Y.")
    return out


st.title("🔮 Forecast")

cities = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

col_top_1, col_top_2 = st.columns([1, 1])

with col_top_1:
    selected_city = st.selectbox("Odaberi grad", cities, index=default_index)

with col_top_2:
    scenario_enabled = st.toggle("Uključi scenario mode", value=True)

st.markdown(
    f"""
    Forecast modul koristi **7-dnevnu vremensku prognozu** i **strict ML model**
    za procjenu buduće razine toplinskog rizika za **{selected_city}**.
    """
)

st.divider()

# --- Scenario controls ---
if scenario_enabled:
    st.markdown("## Scenario mode")
    st.caption(
        "Simuliraj topliji, vlažniji ili manje vjetrovit scenarij kako bi procijenio promjenu toplinskog rizika."
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

# --- Forecast generation ---
try:
    baseline_df = make_ml_forecast(selected_city)
    scenario_df = make_ml_forecast(
        selected_city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
except Exception as exc:
    st.error(f"Forecast nije dostupan: {exc}")
    st.stop()

baseline_peak = baseline_df.sort_values(
    ["heuristic_risk_score", "ml_prediction_confidence"], ascending=[False, False]
).iloc[0]

scenario_peak = scenario_df.sort_values(
    ["heuristic_risk_score", "ml_prediction_confidence"], ascending=[False, False]
).iloc[0]

baseline_high_days = int(
    ((baseline_df["heuristic_risk_level"] == "Visok") | (baseline_df["heuristic_risk_level"] == "Vrlo visok")).sum()
)
scenario_high_days = int(
    ((scenario_df["heuristic_risk_level"] == "Visok") | (scenario_df["heuristic_risk_level"] == "Vrlo visok")).sum()
)

# --- KPI row ---
st.markdown("## Forecast sažetak")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sutrašnji ML rizik", str(baseline_df.iloc[0]["ml_predicted_label"]))
k2.metric("Sutrašnja ML confidence", f"{baseline_df.iloc[0]['ml_prediction_confidence']:.2f}")
k3.metric("Peak baseline heuristic score", f"{baseline_peak['heuristic_risk_score']:.1f}")
k4.metric("Baseline visoki+vrlo visoki dani", baseline_high_days)

if scenario_enabled:
    st.markdown("### Usporedba baseline vs scenario")
    s_k1, s_k2, s_k3, s_k4 = st.columns(4)
    s_k1.metric(
        "Scenario peak score",
        f"{scenario_peak['heuristic_risk_score']:.1f}",
        delta=f"{scenario_peak['heuristic_risk_score'] - baseline_peak['heuristic_risk_score']:.1f}",
    )
    s_k2.metric(
        "Scenario peak apparent temp",
        f"{scenario_peak['apparent_temp_max']:.1f} °C",
        delta=f"{scenario_peak['apparent_temp_max'] - baseline_peak['apparent_temp_max']:.1f} °C",
    )
    s_k3.metric(
        "Scenario visoki+vrlo visoki dani",
        scenario_high_days,
        delta=scenario_high_days - baseline_high_days,
    )
    s_k4.metric(
        "Scenario ML confidence (peak)",
        f"{scenario_peak['ml_prediction_confidence']:.2f}",
    )

st.divider()

# --- Charts ---
left, right = st.columns([1.6, 1])

with left:
    st.markdown("### Baseline vs scenario projected score")

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
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

with right:
    st.markdown("### Peak day interpretacija")

    st.write(
        f"""
        **Baseline peak day:** {pd.to_datetime(baseline_peak['date']).strftime('%d.%m.%Y.')}  
        - heuristic risk: **{baseline_peak['heuristic_risk_level']}**
        - heuristic score: **{baseline_peak['heuristic_risk_score']:.1f}**
        - ML label: **{baseline_peak['ml_predicted_label']}**
        - ML confidence: **{baseline_peak['ml_prediction_confidence']:.2f}**
        """
    )

    if scenario_enabled:
        st.write(
            f"""
            **Scenario peak day:** {pd.to_datetime(scenario_peak['date']).strftime('%d.%m.%Y.')}  
            - heuristic risk: **{scenario_peak['heuristic_risk_level']}**
            - heuristic score: **{scenario_peak['heuristic_risk_score']:.1f}**
            - ML label: **{scenario_peak['ml_predicted_label']}**
            - ML confidence: **{scenario_peak['ml_prediction_confidence']:.2f}**
            """
        )

    city_text, services_text, tourism_text = recommendation_text(str(scenario_peak["heuristic_risk_level"] if scenario_enabled else baseline_peak["heuristic_risk_level"]))

    with st.expander("Preporuke za gradove", expanded=True):
        st.write(city_text)

    with st.expander("Preporuke za javne službe", expanded=True):
        st.write(services_text)

    with st.expander("Preporuke za turizam", expanded=True):
        st.write(tourism_text)

st.divider()

# --- ML class probability view ---
st.markdown("## ML pogled po danima")

proba_cols = [c for c in baseline_df.columns if c.startswith("proba_class_")]
proba_display = baseline_df[["date", "ml_predicted_label", "ml_prediction_confidence"] + proba_cols].copy()
proba_display = to_display_date(proba_display)

st.dataframe(proba_display, use_container_width=True, hide_index=True)

st.divider()

# --- Weather charts ---
st.markdown("## Vremenski signali")

weather_plot_df = baseline_df[
    ["date", "temp_max", "temp_min", "apparent_temp_max"]
].copy().melt(id_vars="date", var_name="metric", value_name="value")

fig_weather = px.line(
    weather_plot_df,
    x="date",
    y="value",
    color="metric",
    markers=True,
    title=f"Baseline forecast signali — {selected_city}",
)
fig_weather.update_layout(
    xaxis_title="Datum",
    yaxis_title="°C",
    margin=dict(l=20, r=20, t=50, b=20),
)
st.plotly_chart(fig_weather, use_container_width=True)

if scenario_enabled:
    st.markdown("### Usporedba baseline vs scenario apparent temperature")
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
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_temp_compare, use_container_width=True)

st.divider()

# --- Forecast tables ---
st.markdown("## Tablični pregled")

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

baseline_table = baseline_df[table_cols].copy()
baseline_table = to_display_date(baseline_table)

tab1, tab2 = st.tabs(["Baseline", "Scenario"])

with tab1:
    st.dataframe(baseline_table, use_container_width=True, hide_index=True)

with tab2:
    scenario_table = scenario_df[table_cols].copy()
    scenario_table = to_display_date(scenario_table)
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

st.divider()

st.markdown("## Tumačenje forecast logike")
st.write(
    """
    Forecast modul kombinira:
    - vremensku prognozu za idućih 7 dana
    - heuristički projected heat risk engine
    - strict ML model za klasifikaciju buduće razine rizika
    - scenario simulation za testiranje osjetljivosti sustava na toplije i nepovoljnije uvjete
    """
)

st.success(
    """
    Ova stranica HeatSafe HR pretvara iz jednostavne prognoze u AI/ML decision-support alat:
    korisnik ne vidi samo vrijeme, nego i kako bi promjena uvjeta mogla povećati toplinski rizik.
    """
)