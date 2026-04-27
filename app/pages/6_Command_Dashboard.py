from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary
from src.forecast_engine import make_ml_forecast
from src.impact_engine import identify_priority_groups
from src.reliability_engine import (
    build_multi_city_reliability_table,
    build_reliability_snapshot,
    build_system_health_summary,
)
from src.resource_routing_engine import (
    build_top_dispatch_summary,
    recommend_dispatch_resources,
)
from src.sidebar import render_app_sidebar

st.set_page_config(page_title="Command Dashboard", page_icon="🧭", layout="wide")

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

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

HEALTH_COLOR_MAP = {
    "Healthy": "#2E8B57",
    "Watch": "#E6A700",
    "Degraded": "#C0392B",
    "No data": "#64748b",
}

CONSENSUS_COLOR_MAP = {
    "Strong consensus": "#2E8B57",
    "Moderate consensus": "#E6A700",
    "Mixed signals": "#E67E22",
    "Low consensus": "#C0392B",
    "Insufficient comparison": "#64748b",
}

CONFIDENCE_COLOR_MAP = {
    "High": "#2E8B57",
    "Moderate": "#E6A700",
    "Low": "#C0392B",
}

READINESS_COLOR_MAP = {
    "Monitoring": "#2E8B57",
    "Prepared": "#E6A700",
    "Elevated Readiness": "#E67E22",
    "Critical Preparedness": "#C0392B",
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1400px;
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
        max-width: 1100px;
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
        margin-bottom: 0.4rem;
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

    .soft-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.72;
        font-size: 0.95rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.88rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .note-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #0f172a;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
        line-height: 1.65;
    }

    .priority-card {
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
        margin-bottom: 0.2rem;
    }

    .small-muted {
        font-size: 0.9rem;
        color: #64748b;
        line-height: 1.6;
    }

    div[data-baseweb="tab-list"] {
        gap: 1.2rem;
        margin-top: 0.9rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
    }

    .stDataFrame {
        border-radius: 14px;
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


def render_list_card(title: str, items: list[str]) -> None:
    safe_items = items if items else ["Nema dostupnih stavki za prikaz."]
    list_html = "".join(f"<li>{item}</li>" for item in safe_items)
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">{title}</div>
            <ul class="soft-list">
                {list_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


@st.cache_data(ttl=1800)
def build_dashboard_reliability_table(
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> pd.DataFrame:
    return build_multi_city_reliability_table(
        cities=CITIES,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )


@st.cache_data(ttl=1800)
def build_selected_city_context(
    city: str,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> dict:
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

    reliability_snapshot = build_reliability_snapshot(
        city=city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )

    priority_groups = identify_priority_groups(
        summary,
        reliability_snapshot["v3_signal"],
    )

    dispatch_df = recommend_dispatch_resources(
        city=city,
        escalation_label=reliability_snapshot["v3_signal"] or "Stable",
        priority_groups=priority_groups,
        top_n=5,
    )

    top_dispatch_summary = build_top_dispatch_summary(dispatch_df)

    return {
        "forecast_df": forecast_df,
        "summary": summary,
        "reliability_snapshot": reliability_snapshot,
        "priority_groups": priority_groups,
        "dispatch_df": dispatch_df,
        "top_dispatch_summary": top_dispatch_summary,
    }


default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = CITIES.index(default_city) if default_city in CITIES else 0

scenario_enabled = st.toggle("Scenario mode za Command Dashboard", value=True)

if scenario_enabled:
    s1, s2, s3 = st.columns(3)
    with s1:
        temperature_delta = st.slider("Promjena temperature (°C)", -2, 12, 6, 1, key="cmd_temp_delta")
    with s2:
        humidity_delta = st.slider("Promjena vlage (%)", -20, 30, 10, 1, key="cmd_humidity_delta")
    with s3:
        wind_delta = st.slider("Promjena vjetra (m/s)", -8, 5, -3, 1, key="cmd_wind_delta")
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

reliability_df = build_dashboard_reliability_table(
    temperature_delta=temperature_delta,
    humidity_delta=humidity_delta,
    wind_delta=wind_delta,
)

system_health = build_system_health_summary(reliability_df)

selected_city = st.selectbox("Odaberi grad", CITIES, index=default_index)
st.session_state.selected_city = selected_city

selected_context = build_selected_city_context(
    city=selected_city,
    temperature_delta=temperature_delta,
    humidity_delta=humidity_delta,
    wind_delta=wind_delta,
)
selected_snapshot = selected_context["reliability_snapshot"]
selected_summary = selected_context["summary"]
dispatch_df = selected_context["dispatch_df"]
top_dispatch_summary = selected_context["top_dispatch_summary"]
priority_groups = selected_context["priority_groups"]

render_app_sidebar(
    selected_city=selected_city,
    risk_level=selected_snapshot.get("next_24h_risk"),
    readiness_status=selected_snapshot.get("readiness_status"),
    escalation_label=selected_snapshot.get("v3_signal"),
)

priority_df = reliability_df.sort_values(
    ["impact_adjusted_priority", "reliability_score"],
    ascending=[False, False],
).reset_index(drop=True)

uncertainty_df = reliability_df[
    (reliability_df["operator_review_required"] == True)
    | (reliability_df["consensus_status"].isin(["Low consensus", "Mixed signals"]))
].copy()

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🧭 Command Dashboard</div>
        <div class="page-hero-subtitle">
            Operator cockpit za više gradova odjednom. Ovdje se spajaju impact-adjusted priority,
            model consensus, confidence level, uncertainty warning i dispatch routing spreman za odluke.
            Cilj nije samo vidjeti koji je grad topliji, nego gdje je signal najvažniji i koliko je metodološki pouzdan.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card(
        "System Health",
        system_health["system_health"],
        "Reliability core",
    )
with m2:
    metric_card(
        "Avg reliability",
        fmt_float(system_health["avg_reliability_score"], 1),
        "Across cities",
    )
with m3:
    metric_card(
        "Strong consensus",
        str(system_health["strong_consensus_count"]),
        "Cities aligned",
    )
with m4:
    metric_card(
        "Review required",
        str(system_health["operator_review_count"]),
        "Operator queue",
    )
with m5:
    top_priority_city = priority_df.iloc[0]["city"] if not priority_df.empty else "N/A"
    metric_card(
        "Top priority city",
        str(top_priority_city),
        "Impact-adjusted priority",
    )

health_color = HEALTH_COLOR_MAP.get(system_health["system_health"], "#64748b")
st.markdown(
    pill(system_health["system_health"], health_color),
    unsafe_allow_html=True,
)

strict_warning_note = ""
if reliability_df["strict_model_available"].eq(False).any():
    strict_warning_note = (
        "Strict model inference trenutno nije dostupan, pa je reliability layer djelomično "
        "ograničen na v1 + v3 + consensus heuristiku. To nije crash, nego graceful fallback."
    )

st.markdown(
    f"""
    <div class="note-box">
        <b>System interpretation:</b> dashboard sada prati ne samo toplinski rizik, nego i
        <b>pouzdanost modelnog signala</b>. Ako se modeli ne slažu ili je confidence nizak,
        sustav automatski označava potrebu za ljudskom provjerom.
        {"<br><br><b>Napomena:</b> " + strict_warning_note if strict_warning_note else ""}
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Operator cockpit",
        "Consensus & reliability",
        "Priority ranking",
    ]
)

with tabs[0]:
    st.markdown("## Selected city operator cockpit")

    badge_html = (
        pill(
            selected_snapshot["readiness_status"],
            READINESS_COLOR_MAP.get(selected_snapshot["readiness_status"], "#64748b"),
        )
        + pill(
            selected_snapshot["consensus_status"],
            CONSENSUS_COLOR_MAP.get(selected_snapshot["consensus_status"], "#64748b"),
        )
        + pill(
            selected_snapshot["confidence_level"],
            CONFIDENCE_COLOR_MAP.get(selected_snapshot["confidence_level"], "#64748b"),
        )
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Impact-adjusted priority",
            fmt_float(selected_snapshot["impact_adjusted_priority"], 1),
            "Heat + escalation + vulnerability",
        )
    with c2:
        metric_card(
            "Reliability score",
            fmt_float(selected_snapshot["reliability_score"], 1),
            selected_snapshot["confidence_level"],
        )
    with c3:
        metric_card(
            "Next 7d peak",
            selected_snapshot["next_7d_peak_level"],
            fmt_float(selected_snapshot["next_7d_peak_score"], 1),
        )
    with c4:
        metric_card(
            "Operator review",
            "YES" if selected_snapshot["operator_review_required"] else "NO",
            "Human-in-the-loop flag",
        )

    o1, o2 = st.columns([1.1, 1])

    with o1:
        render_text_card(
            "Model consensus summary",
            f"""
            <b>v1 signal:</b> {selected_snapshot["v1_signal"] or "N/A"}<br>
            <b>v2 signal:</b> {selected_snapshot["v2_signal"] or "N/A"}<br>
            <b>v3 signal:</b> {selected_snapshot["v3_signal"] or "N/A"}<br>
            <b>Consensus:</b> {selected_snapshot["consensus_status"]}<br>
            <b>Confidence level:</b> {selected_snapshot["confidence_level"]}<br>
            <b>Uncertainty warning:</b> {selected_snapshot["uncertainty_warning"]}
            """,
        )

    with o2:
        render_text_card(
            "Operational context",
            f"""
            <b>Readiness:</b> {selected_snapshot["readiness_status"]}<br>
            <b>Next 24h risk:</b> {selected_snapshot["next_24h_risk"]}<br>
            <b>Vulnerability band:</b> {selected_snapshot["vulnerability_band"]}<br>
            <b>Vulnerability index:</b> {fmt_float(selected_snapshot["vulnerability_index"], 1)}<br>
            <b>Priority groups:</b> {", ".join(priority_groups) if priority_groups else "N/A"}<br>
            <b>Top dispatch summary:</b><br>{top_dispatch_summary}
            """,
        )

    st.markdown("### Top dispatch resource")
    if dispatch_df.empty:
        st.info("Nema dispatch preporuka za ovaj grad.")
    else:
        top_dispatch_row = dispatch_df.iloc[0]

        d1, d2, d3 = st.columns(3)
        with d1:
            metric_card(
                "Dispatch resource",
                str(top_dispatch_row["resource_name"]),
                str(top_dispatch_row["resource_type"]),
            )
        with d2:
            metric_card(
                "Dispatch score",
                fmt_float(top_dispatch_row["dispatch_score"], 1),
                "Routing score",
            )
        with d3:
            distance_text = (
                f"{float(top_dispatch_row['nearest_critical_distance_km']):.2f} km"
                if pd.notna(top_dispatch_row["nearest_critical_distance_km"])
                else "N/A"
            )
            metric_card(
                "Nearest critical point",
                str(top_dispatch_row["nearest_critical_point"]),
                distance_text,
            )

        st.info(top_dispatch_summary)

    if selected_snapshot["operator_review_required"]:
        st.error(selected_snapshot["uncertainty_warning"])
    else:
        st.success("No major uncertainty warning for current selected city context.")

with tabs[1]:
    st.markdown("## Consensus & reliability across cities")

    review_count = int(reliability_df["operator_review_required"].sum())
    low_conf_count = int((reliability_df["confidence_level"] == "Low").sum())
    mixed_count = int(reliability_df["consensus_status"].isin(["Mixed signals", "Low consensus"]).sum())

    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Operator review queue", str(review_count), "Cities requiring human check")
    with r2:
        metric_card("Low confidence", str(low_conf_count), "Potential reliability concern")
    with r3:
        metric_card("Mixed / low consensus", str(mixed_count), "Disagreement between models")

    consensus_display_df = reliability_df[
        [
            "city",
            "v1_signal",
            "v2_signal",
            "v3_signal",
            "consensus_status",
            "confidence_level",
            "reliability_score",
            "uncertainty_warning",
            "operator_review_required",
        ]
    ].copy()

    st.dataframe(consensus_display_df, use_container_width=True, hide_index=True)

    if uncertainty_df.empty:
        st.success("Trenutno nema gradova u operator review queue.")
    else:
        st.markdown("### Operator review queue")

        review_display_df = uncertainty_df[
            [
                "city",
                "consensus_status",
                "confidence_level",
                "impact_adjusted_priority",
                "reliability_score",
                "uncertainty_warning",
                "operator_review_required",
            ]
        ].copy()

        review_display_df = review_display_df.sort_values(
            ["impact_adjusted_priority", "reliability_score"],
            ascending=[False, True],
        )

        st.dataframe(review_display_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("## Priority ranking")

    top_cols = st.columns(min(3, len(priority_df)))
    for i, (_, row) in enumerate(priority_df.head(3).iterrows()):
        with top_cols[i]:
            consensus_color = CONSENSUS_COLOR_MAP.get(row["consensus_status"], "#64748b")
            confidence_color = CONFIDENCE_COLOR_MAP.get(row["confidence_level"], "#64748b")

            st.markdown(
                f"""
                <div class="priority-card">
                    <div class="mini-title">Priority rank #{i+1}</div>
                    <div class="big-city">{row['city']}</div>
                    <div style="margin:0.35rem 0 0.55rem 0;">
                        <span class="status-pill" style="background:{consensus_color};">{row['consensus_status']}</span>
                        <span class="status-pill" style="background:{confidence_color};">{row['confidence_level']}</span>
                    </div>
                    <div class="small-muted">Impact-adjusted priority: <b>{float(row['impact_adjusted_priority']):.1f}</b></div>
                    <div class="small-muted">Readiness: <b>{row['readiness_status']}</b></div>
                    <div class="small-muted">Next 7d peak: <b>{row['next_7d_peak_level']} ({float(row['next_7d_peak_score']):.1f})</b></div>
                    <div class="small-muted">v3 signal: <b>{row['v3_signal']}</b></div>
                    <div class="small-muted">Reliability score: <b>{float(row['reliability_score']):.1f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Full city ranking")
    ranking_df = priority_df[
        [
            "city",
            "impact_adjusted_priority",
            "readiness_status",
            "next_7d_peak_level",
            "next_7d_peak_score",
            "vulnerability_band",
            "v3_signal",
            "consensus_status",
            "confidence_level",
            "reliability_score",
            "operator_review_required",
        ]
    ].copy()

    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note-box">
            Prioritet više nije određen samo temperaturom. Rang sada kombinira toplinski peak,
            escalation signal i socio-ekonomsku ranjivost, a uz to dodatno provjerava koliko je
            taj modelni signal metodološki pouzdan. Time dashboard postaje stvarni operator cockpit,
            a ne samo pregled vremenskih uvjeta.
        </div>
        """,
        unsafe_allow_html=True,
    )