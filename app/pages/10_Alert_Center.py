from __future__ import annotations

import sys
from pathlib import Path

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

st.set_page_config(page_title="Alert Center", page_icon="🚨", layout="wide")

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
        line-height: 1.55;
        opacity: 0.95;
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
        margin-bottom: 0.7rem;
    }

    .panel-text {
        color: #334155;
        line-height: 1.65;
        font-size: 0.95rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.72rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.86rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
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
        gap: 1.2rem;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
    }
    </style>
    """,
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


def pill(text: str, color: str) -> str:
    return f'<span class="status-pill" style="background:{color};">{text}</span>'


@st.cache_data(ttl=1800)
def load_city_snapshot(city: str, temperature_delta: float, humidity_delta: float, wind_delta: float):
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


st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🚨 Alert Center</div>
        <div class="page-hero-subtitle">
            Operator-focused modul za izdavanje i praćenje alerta. Stranica dodaje severity rules,
            alert log, operator summary, exportable alert packages i smart communication layer
            za više gradova odjednom.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
packages = {}

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

issued_count = int((snapshot_df["alert_issued"] == "Yes").sum())
warning_plus = int(snapshot_df["alert_severity"].isin(["Heat Warning", "Critical Alert"]).sum())
critical_count = int((snapshot_df["alert_severity"] == "Critical Alert").sum())
top_city = snapshot_df.iloc[0]["city"] if not snapshot_df.empty else "N/A"
likely_escalation_count = int((snapshot_df["escalation_label_72h"] == "Likely escalation").sum())

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
    metric_card("Top operator priority", str(top_city), "Most urgent city")

tabs = st.tabs(
    [
        "Current snapshot",
        "Alert history",
        "Alert packages",
        "Communication layer",
    ]
)

with tabs[0]:
    st.markdown("### Current operator snapshot")

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
        margin=dict(l=20, r=20, t=50, b=20),
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

with tabs[1]:
    st.markdown("### Alert history")

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
    st.markdown("### Exportable alert packages")

    default_city = st.session_state.get("selected_city", DEFAULT_CITY)
    package_city = st.selectbox(
        "Odaberi grad za alert package",
        CITIES,
        index=CITIES.index(default_city) if default_city in CITIES else 0,
        key="package_city_alert_center",
    )
    st.session_state.selected_city = package_city

    package_text = packages[package_city]
    selected_row = snapshot_df[snapshot_df["city"] == package_city].iloc[0]

    badge_html = pill(
        selected_row["alert_severity"],
        ALERT_COLOR_MAP.get(selected_row["alert_severity"], "#64748b"),
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    if pd.notna(selected_row.get("escalation_label_72h")):
        escalation_badge = pill(
            selected_row["escalation_label_72h"],
            ESCALATION_COLOR_MAP.get(selected_row["escalation_label_72h"], "#64748b"),
        )
        st.markdown(escalation_badge, unsafe_allow_html=True)

    st.code(package_text, language="text")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download alert package (.txt)",
            data=package_text.encode("utf-8"),
            file_name=f"heatsafe_hr_alert_package_{package_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_alert_package_{package_city}",
        )
    with c2:
        st.download_button(
            "⬇ Download operator row (.csv)",
            data=csv_bytes(snapshot_df[snapshot_df["city"] == package_city]),
            file_name=f"heatsafe_hr_operator_row_{package_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_operator_row_{package_city}",
        )

with tabs[3]:
    st.markdown("### Smart Alerting communication layer")

    if snapshot_df.empty:
        st.info("Nema alert podataka za generiranje komunikacijskih poruka.")
    else:
        comm_default_city = st.session_state.get("selected_city", DEFAULT_CITY)
        selected_comm_city = st.selectbox(
            "Odaberi grad za komunikacijski paket",
            CITIES,
            index=CITIES.index(comm_default_city) if comm_default_city in CITIES else 0,
            key="selected_comm_city_alert_center",
        )

        selected_alert_row = snapshot_df[snapshot_df["city"] == selected_comm_city].iloc[0]
        comm_package = build_alert_communication_package(selected_alert_row.to_dict())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Alert severity", selected_alert_row["alert_severity"], "Public communication level")
        with c2:
            metric_card("Readiness", selected_alert_row["readiness"], "Operational posture")
        with c3:
            metric_card(
                "72h escalation",
                selected_alert_row["escalation_label_72h"],
                f"{selected_alert_row['escalation_probability_72h']:.2f}",
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

        st.markdown("### Download communication package")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                "⬇ Download HR public advisory (.txt)",
                data=comm_package["public_advisory_hr"].encode("utf-8"),
                file_name=f"heatsafe_hr_public_advisory_{selected_comm_city}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_public_hr_{selected_comm_city}",
            )

        with d2:
            st.download_button(
                "⬇ Download EN tourist advisory (.txt)",
                data=comm_package["tourist_advisory_en"].encode("utf-8"),
                file_name=f"heatsafe_hr_tourist_advisory_{selected_comm_city}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_tourist_en_{selected_comm_city}",
            )

        with d3:
            package_text = (
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
                data=package_text.encode("utf-8"),
                file_name=f"heatsafe_hr_communication_package_{selected_comm_city}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_comm_package_{selected_comm_city}",
            )