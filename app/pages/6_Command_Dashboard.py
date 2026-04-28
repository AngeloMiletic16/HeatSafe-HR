from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

st.set_page_config(
    page_title="Command Dashboard | HeatSafe HR",
    page_icon="🧭",
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

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

SYSTEM_STATUS_COLOR_MAP = {
    "Stable": "#2E8B57",
    "Watch": "#E6A700",
    "Degraded": "#C0392B",
    "Fallback mode": "#64748b",
    "No data": "#64748b",
}

CONSENSUS_COLOR_MAP = {
    "Strong consensus": "#0f766e",
    "Moderate consensus": "#2563eb",
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
        padding-top: 1.65rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .page-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
        border-radius: 22px;
        padding: 1.4rem 1.55rem 1.22rem 1.55rem;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.16);
    }

    .page-hero-title {
        font-size: 2.02rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .page-hero-subtitle {
        font-size: 0.99rem;
        line-height: 1.6;
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
        font-size: 0.87rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .section-title {
        font-size: 1.38rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.35rem 0 0.9rem 0;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        min-height: 114px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.42rem;
    }

    .metric-value {
        font-size: 1.72rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.12;
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
        font-size: 1.06rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.7rem;
    }

    .panel-text {
        color: #334155;
        line-height: 1.66;
        font-size: 0.95rem;
    }

    .soft-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.72;
        font-size: 0.94rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.76rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.87rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .note-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #0f172a;
        margin-top: 0.55rem;
        margin-bottom: 0.85rem;
        line-height: 1.66;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #7c2d12;
        margin-top: 0.55rem;
        margin-bottom: 0.85rem;
        line-height: 1.66;
    }

    .success-box {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #065f46;
        margin-top: 0.55rem;
        margin-bottom: 0.85rem;
        line-height: 1.66;
    }

    .priority-card {
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(15,23,42,0.08);
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        background: #ffffff;
        height: 100%;
    }

    .mini-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .big-city {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.22rem;
    }

    .small-muted {
        font-size: 0.91rem;
        color: #64748b;
        line-height: 1.6;
    }

    .context-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.45rem 1.25rem;
    }

    .context-item {
        color: #334155;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    div[data-baseweb="tab-list"] {
        gap: 1rem;
        margin-top: 0.9rem;
        margin-bottom: 0.85rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
    }

    .stDataFrame, .stPlotlyChart {
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


def fmt_float(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def derive_system_status_label(system_health_raw: str, strict_available_any: bool) -> str:
    if system_health_raw == "No data":
        return "No data"
    if not strict_available_any:
        if system_health_raw == "Degraded":
            return "Degraded"
        return "Fallback mode"
    if system_health_raw == "Healthy":
        return "Stable"
    if system_health_raw == "Watch":
        return "Watch"
    if system_health_raw == "Degraded":
        return "Degraded"
    return "Watch"


def build_rank_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    try:
        peak_score = float(row.get("next_7d_peak_score", 0))
        vuln_band = str(row.get("vulnerability_band", ""))
        v3_signal = str(row.get("v3_signal", ""))
        confidence = str(row.get("confidence_level", ""))

        if peak_score >= 50:
            reasons.append("higher forecast peak")
        elif peak_score >= 25:
            reasons.append("elevated forecast peak")

        if vuln_band and vuln_band.lower() not in {"n/a", "low vulnerability"}:
            reasons.append("vulnerability weighting")

        if v3_signal == "Likely escalation":
            reasons.append("72h escalation signal")
        elif v3_signal == "Watch":
            reasons.append("watch escalation signal")

        if confidence == "Low":
            reasons.append("lower signal confidence")
    except Exception:
        pass

    if not reasons:
        return "balanced priority score"

    return ", ".join(reasons[:2])


@st.cache_data(ttl=1800)
def build_dashboard_reliability_table(
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> pd.DataFrame:
    df = build_multi_city_reliability_table(
        cities=CITIES,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
    return df.copy()


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
        reliability_snapshot.get("v3_signal"),
    )

    dispatch_df = recommend_dispatch_resources(
        city=city,
        escalation_label=reliability_snapshot.get("v3_signal") or "Stable",
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

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🧭 Command Dashboard</div>
        <div class="page-hero-subtitle">
            Operator cockpit za više gradova odjednom. Ovdje se spajaju impact-adjusted priority,
            model consensus, confidence level, uncertainty warning i dispatch routing spreman za odluke.
            Cilj nije samo vidjeti koji je grad topliji, nego gdje je signal najvažniji i koliko je metodološki pouzdan.
        </div>
        <div class="chip-row">
            <span class="chip">Operator Cockpit</span>
            <span class="chip">Consensus Layer</span>
            <span class="chip">Reliability</span>
            <span class="chip">Priority Ranking</span>
            <span class="chip">Dispatch Routing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([1.15, 0.85])
with top_left:
    selected_city = st.selectbox("Odaberi grad", CITIES, index=default_index)
    st.session_state.selected_city = selected_city
with top_right:
    scenario_enabled = st.toggle("Scenario mode za Command Dashboard", value=True)

if scenario_enabled:
    with st.expander("Scenario settings", expanded=False):
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

if reliability_df.empty:
    st.warning("Nema dostupnih podataka za Command Dashboard.")
    st.stop()

system_health = build_system_health_summary(reliability_df)

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

strict_available_any = bool(reliability_df["strict_model_available"].fillna(False).any()) if "strict_model_available" in reliability_df.columns else True
system_status_label = derive_system_status_label(system_health.get("system_health", "Watch"), strict_available_any)

render_app_sidebar(
    selected_city=selected_city,
    risk_level=selected_snapshot.get("next_24h_risk"),
    readiness_status=selected_snapshot.get("readiness_status"),
    escalation_label=selected_snapshot.get("v3_signal"),
    escalation_probability=selected_snapshot.get("escalation_probability_72h"),
)

priority_df = reliability_df.sort_values(
    ["impact_adjusted_priority", "reliability_score"],
    ascending=[False, False],
).reset_index(drop=True)

uncertainty_df = reliability_df[
    (reliability_df["operator_review_required"].apply(to_bool))
    | (reliability_df["consensus_status"].isin(["Low consensus", "Mixed signals"]))
].copy()

top_priority_city = priority_df.iloc[0]["city"] if not priority_df.empty else "N/A"
system_status_color = SYSTEM_STATUS_COLOR_MAP.get(system_status_label, "#64748b")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("System status", system_status_label, "Reliability layer")
with m2:
    metric_card("Avg reliability", fmt_float(system_health.get("avg_reliability_score"), 1), "Across cities")
with m3:
    metric_card("Strong consensus", str(system_health.get("strong_consensus_count", 0)), "Cities aligned")
with m4:
    metric_card("Review required", str(system_health.get("operator_review_count", 0)), "Operator queue")
with m5:
    metric_card("Top priority city", str(top_priority_city), "Impact-adjusted priority")

st.markdown(pill(system_status_label, system_status_color), unsafe_allow_html=True)

selected_city_reason = build_rank_reason(
    priority_df[priority_df["city"] == selected_city].iloc[0]
    if selected_city in priority_df["city"].tolist()
    else priority_df.iloc[0]
)

st.markdown(
    f"""
    <div class="note-box">
        <b>Operator interpretation:</b> dashboard danas ne prikazuje samo toplinski signal, nego i
        <b>koliko je taj signal pouzdan za odluku</b>. Za fokusirani grad <b>{selected_city}</b>
        trenutačni readiness je <b>{selected_snapshot.get("readiness_status", "N/A")}</b>,
        next 24h risk je <b>{selected_snapshot.get("next_24h_risk", "N/A")}</b>,
        a prioritet je pojačan kroz <b>{selected_city_reason}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

if not strict_available_any:
    st.markdown(
        """
        <div class="warning-box">
            <b>Fallback mode active:</b> strict model inference trenutačno nije dostupan.
            Dashboard zato koristi graceful fallback kombinaciju <b>v1 + v3 + heuristic consensus</b>.
            To nije crash, ali reliability layer treba čitati kao djelomično ograničen.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="success-box">
            <b>Model stack status:</b> svi ključni slojevi za consensus i reliability su dostupni.
            Dashboard se može čitati kao puni operator cockpit za prioritizaciju gradova.
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
    st.markdown('<div class="section-title">Selected city operator cockpit</div>', unsafe_allow_html=True)

    badge_html = (
        pill(
            selected_snapshot.get("readiness_status", "N/A"),
            READINESS_COLOR_MAP.get(selected_snapshot.get("readiness_status", "N/A"), "#64748b"),
        )
        + pill(
            selected_snapshot.get("consensus_status", "N/A"),
            CONSENSUS_COLOR_MAP.get(selected_snapshot.get("consensus_status", "N/A"), "#64748b"),
        )
        + pill(
            selected_snapshot.get("confidence_level", "N/A"),
            CONFIDENCE_COLOR_MAP.get(selected_snapshot.get("confidence_level", "N/A"), "#64748b"),
        )
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Impact-adjusted priority",
            fmt_float(selected_snapshot.get("impact_adjusted_priority"), 1),
            "Heat + escalation + vulnerability",
        )
    with c2:
        metric_card(
            "Reliability score",
            fmt_float(selected_snapshot.get("reliability_score"), 1),
            selected_snapshot.get("confidence_level", "N/A"),
        )
    with c3:
        metric_card(
            "Next 7d peak",
            str(selected_snapshot.get("next_7d_peak_level", "N/A")),
            fmt_float(selected_snapshot.get("next_7d_peak_score"), 1),
        )
    with c4:
        metric_card(
            "Operator review",
            "YES" if to_bool(selected_snapshot.get("operator_review_required")) else "NO",
            "Human-in-the-loop flag",
        )

    o1, o2 = st.columns([1, 1])

    with o1:
        render_text_card(
            "Model consensus summary",
            f"""
            <div class="context-grid">
                <div class="context-item"><b>v1 signal:</b> {selected_snapshot.get("v1_signal") or "N/A"}</div>
                <div class="context-item"><b>v2 signal:</b> {selected_snapshot.get("v2_signal") or "N/A"}</div>
                <div class="context-item"><b>v3 signal:</b> {selected_snapshot.get("v3_signal") or "N/A"}</div>
                <div class="context-item"><b>Consensus:</b> {selected_snapshot.get("consensus_status", "N/A")}</div>
                <div class="context-item"><b>Confidence:</b> {selected_snapshot.get("confidence_level", "N/A")}</div>
                <div class="context-item"><b>Rank reason:</b> {selected_city_reason}</div>
            </div>
            <br>
            <b>Uncertainty warning:</b> {selected_snapshot.get("uncertainty_warning", "N/A")}
            """,
        )

    with o2:
        render_text_card(
            "Operational context",
            f"""
            <div class="context-grid">
                <div class="context-item"><b>Readiness:</b> {selected_snapshot.get("readiness_status", "N/A")}</div>
                <div class="context-item"><b>Next 24h risk:</b> {selected_snapshot.get("next_24h_risk", "N/A")}</div>
                <div class="context-item"><b>Vulnerability band:</b> {selected_snapshot.get("vulnerability_band", "N/A")}</div>
                <div class="context-item"><b>Vulnerability index:</b> {fmt_float(selected_snapshot.get("vulnerability_index"), 1)}</div>
                <div class="context-item"><b>72h escalation:</b> {selected_snapshot.get("v3_signal", "N/A")}</div>
                <div class="context-item"><b>Escalation prob.:</b> {fmt_float(selected_snapshot.get("escalation_probability_72h"), 2)}</div>
            </div>
            <br>
            <b>Priority groups:</b> {", ".join(priority_groups) if priority_groups else "N/A"}
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

        st.markdown(
            f"""
            <div class="note-box">
                <b>Dispatch interpretation:</b> {top_dispatch_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if to_bool(selected_snapshot.get("operator_review_required")):
        st.error(str(selected_snapshot.get("uncertainty_warning", "Operator review preporučen.")))
    else:
        st.success("Za trenutno odabrani grad nema kritičnog metodološkog upozorenja.")

with tabs[1]:
    st.markdown('<div class="section-title">Consensus & reliability across cities</div>', unsafe_allow_html=True)

    review_count = int(reliability_df["operator_review_required"].apply(to_bool).sum())
    low_conf_count = int((reliability_df["confidence_level"] == "Low").sum())
    mixed_count = int(reliability_df["consensus_status"].isin(["Mixed signals", "Low consensus"]).sum())

    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Operator review queue", str(review_count), "Cities requiring human check")
    with r2:
        metric_card("Low confidence", str(low_conf_count), "Potential reliability concern")
    with r3:
        metric_card("Mixed / low consensus", str(mixed_count), "Disagreement between models")

    findings_left, findings_right = st.columns([1, 1])
    with findings_left:
        render_list_card(
            "Operational findings",
            [
                f"Cities in review queue: {review_count}",
                f"Cities with low confidence: {low_conf_count}",
                f"Cities with mixed or low consensus: {mixed_count}",
                f"Strict model available somewhere: {'Yes' if strict_available_any else 'No'}",
            ],
        )
    with findings_right:
        render_list_card(
            "How to read this tab",
            [
                "Consensus status govori slažu li se modelni slojevi.",
                "Confidence level govori koliko je signal stabilan za operativnu odluku.",
                "Operator review flag označava gdje je potreban dodatni ljudski pogled.",
            ],
        )

    consensus_display_df = reliability_df[
        [
            "city",
            "v1_signal",
            "v2_signal",
            "v3_signal",
            "consensus_status",
            "confidence_level",
            "reliability_score",
            "operator_review_required",
            "uncertainty_warning",
        ]
    ].copy()

    consensus_display_df["reliability_score"] = consensus_display_df["reliability_score"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)
    consensus_display_df["operator_review_required"] = consensus_display_df["operator_review_required"].apply(lambda x: "Yes" if to_bool(x) else "No")

    st.dataframe(consensus_display_df, use_container_width=True, hide_index=True)

    if uncertainty_df.empty:
        st.markdown(
            """
            <div class="success-box">
                Trenutno nema gradova u operator review queue.
            </div>
            """,
            unsafe_allow_html=True,
        )
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
            ]
        ].copy()

        review_display_df = review_display_df.sort_values(
            ["impact_adjusted_priority", "reliability_score"],
            ascending=[False, True],
        )
        review_display_df["impact_adjusted_priority"] = review_display_df["impact_adjusted_priority"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)
        review_display_df["reliability_score"] = review_display_df["reliability_score"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)

        st.dataframe(review_display_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown('<div class="section-title">Priority ranking</div>', unsafe_allow_html=True)

    top_n = min(3, len(priority_df))
    top_cols = st.columns(top_n)

    for i, (_, row) in enumerate(priority_df.head(top_n).iterrows()):
        with top_cols[i]:
            consensus_color = CONSENSUS_COLOR_MAP.get(row["consensus_status"], "#64748b")
            confidence_color = CONFIDENCE_COLOR_MAP.get(row["confidence_level"], "#64748b")
            rank_reason = build_rank_reason(row)

            st.markdown(
                f"""
                <div class="priority-card">
                    <div class="mini-title">Priority rank #{i + 1}</div>
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
                    <div class="small-muted">Rank reason: <b>{rank_reason}</b></div>
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

    ranking_df["impact_adjusted_priority"] = ranking_df["impact_adjusted_priority"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)
    ranking_df["next_7d_peak_score"] = ranking_df["next_7d_peak_score"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)
    ranking_df["reliability_score"] = ranking_df["reliability_score"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)
    ranking_df["operator_review_required"] = ranking_df["operator_review_required"].apply(lambda x: "Yes" if to_bool(x) else "No")
    ranking_df["rank_reason"] = priority_df.apply(build_rank_reason, axis=1)

    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note-box">
            Prioritet više nije određen samo temperaturom. Rang sada kombinira toplinski peak,
            escalation signal i socio-ekonomsku ranjivost, a zatim dodatno provjerava koliko je
            taj modelni signal metodološki pouzdan. Time dashboard postaje pravi operator cockpit,
            a ne samo pregled vremenskih uvjeta.
        </div>
        """,
        unsafe_allow_html=True,
    )