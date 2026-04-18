from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.decision_engine import (
    assess_event_risk,
    build_city_readiness_summary,
    build_escalation_plan,
    build_event_brief,
    build_executive_brief,
    build_scenario_comparison_brief,
    build_sector_actions,
    readiness_to_color,
    risk_to_color,
)
from src.forecast_engine import make_ml_forecast


RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1350px;
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
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 0.2rem;
        word-break: break-word;
    }

    .metric-sub {
        font-size: 0.88rem;
        color: #64748b;
    }

    .status-pill {
        display: inline-block;
        padding: 0.42rem 0.82rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.94rem;
    }

    .soft-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        height: 100%;
    }

    .panel-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }

    .panel-text {
        color: #334155;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    .soft-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .report-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        margin-top: 0.8rem;
    }

    .subsection-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.25rem 0 0.75rem 0;
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

    div.stDownloadButton > button:focus:not(:active) {
        border: none;
        color: white;
    }

    div[data-baseweb="tab-list"] {
        gap: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
        margin-right: 0.25rem;
    }

    div[data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(text: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{text}</span>',
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


def render_list_card(title: str, items: list[str]) -> None:
    list_html = "".join(f"<li>{item}</li>" for item in items)
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


def to_display_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column]).dt.strftime("%d.%m.%Y.")
    return out


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_timeline_figure(df: pd.DataFrame, city_name: str) -> go.Figure:
    plot_df = df.copy().sort_values("date").reset_index(drop=True)
    plot_df["date_str"] = pd.to_datetime(plot_df["date"]).dt.strftime("%d.%m.%Y.")

    colors = [RISK_COLOR_MAP.get(level, "#64748b") for level in plot_df["heuristic_risk_level"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["date_str"],
            y=plot_df["heuristic_risk_score"],
            marker_color=colors,
            text=plot_df["heuristic_risk_score"].round(0).astype(int).astype(str),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Risk level: %{customdata[0]}<br>"
                "Risk score: %{y:.1f}<br>"
                "ML label: %{customdata[1]}<br>"
                "ML confidence: %{customdata[2]:.2f}<br>"
                "Temp max: %{customdata[3]:.1f} °C<br>"
                "Apparent temp max: %{customdata[4]:.1f} °C<br>"
                "Humidity mean: %{customdata[5]:.1f} %<extra></extra>"
            ),
            customdata=plot_df[
                [
                    "heuristic_risk_level",
                    "ml_predicted_label",
                    "ml_prediction_confidence",
                    "temp_max",
                    "apparent_temp_max",
                    "humidity_mean",
                ]
            ].values,
            name="Projected risk score",
        )
    )

    for y, label, color in [
        (25, "Umjeren", "#E6A700"),
        (50, "Visok", "#E67E22"),
        (75, "Vrlo visok", "#C0392B"),
    ]:
        fig.add_hline(
            y=y,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            opacity=0.55,
            annotation_text=label,
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"Operational risk timeline — {city_name}",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        bargap=0.18,
    )

    fig.update_xaxes(
        title_text="Datum",
        showgrid=False,
        tickangle=0,
    )

    fig.update_yaxes(
        title_text="Projected Heat Risk Score",
        range=[0, 100],
        showgrid=True,
        gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
    )

    return fig


st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🚨 Action Center / Alert Center</div>
        <div class="page-hero-subtitle">
            Operativni command center za toplinski rizik. Ova stranica spaja forecast,
            readiness status, sektor-specifične preporuke, event risk procjenu i export-ready briefove.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cities = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]
default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = cities.index(default_city) if default_city in cities else 0

top1, top2 = st.columns([1, 1])

with top1:
    selected_city = st.selectbox("Odaberi grad", cities, index=default_index)
    st.session_state.selected_city = selected_city

with top2:
    scenario_enabled = st.toggle("Scenario mode", value=True)

if scenario_enabled:
    s1, s2, s3 = st.columns(3)
    with s1:
        temperature_delta = st.slider("Promjena temperature (°C)", -2, 12, 6, 1)
    with s2:
        humidity_delta = st.slider("Promjena vlage (%)", -20, 30, 10, 1)
    with s3:
        wind_delta = st.slider("Promjena vjetra (m/s)", -8, 5, -3, 1)
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

try:
    baseline_df = make_ml_forecast(selected_city)
    scenario_df = make_ml_forecast(
        selected_city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
except Exception as exc:
    st.error(f"Action Center nije dostupan: {exc}")
    st.stop()

active_df = scenario_df if scenario_enabled else baseline_df
summary = build_city_readiness_summary(selected_city, active_df)
actions = build_sector_actions(summary["next_7d_peak_level"])
escalation = build_escalation_plan(summary["next_7d_peak_level"])

executive_brief = build_executive_brief(selected_city, active_df, scenario_used=scenario_enabled)
scenario_brief = build_scenario_comparison_brief(
    selected_city,
    baseline_df,
    scenario_df,
    temperature_delta=temperature_delta,
    humidity_delta=humidity_delta,
    wind_delta=wind_delta,
)

st.markdown("## City Readiness Status")
badge(summary["readiness_status"], readiness_to_color(summary["readiness_status"]))

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Next 24h risk", summary["next_24h_level"])
with m2:
    metric_card("Next 24h score", f"{summary['next_24h_score']:.1f}")
with m3:
    metric_card("Next 72h peak", summary["next_72h_peak_level"])
with m4:
    metric_card("Next 7d peak", summary["next_7d_peak_level"])
with m5:
    metric_card("High-risk days", str(summary["high_risk_days"]))

st.info(
    f"""
    **Operational status for {selected_city}:** {summary['readiness_status']}  
    Peak risk within the next 7 days is expected on **{summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}**
    with score **{summary['next_7d_peak_score']:.1f}**.
    """
)

tabs = st.tabs(
    [
        "Action Center",
        "Event / Tourism Risk Check",
        "Executive Summary",
    ]
)

with tabs[0]:
    st.markdown("### Operational forecast timeline")

    peak_row = active_df.sort_values(
        ["heuristic_risk_score", "apparent_temp_max", "ml_prediction_confidence"],
        ascending=[False, False, False],
    ).iloc[0]

    trend_delta = float(active_df.iloc[-1]["heuristic_risk_score"] - active_df.iloc[0]["heuristic_risk_score"])

    g1, g2, g3 = st.columns(3)
    with g1:
        metric_card(
            "Peak day",
            pd.to_datetime(peak_row["date"]).strftime("%d.%m.%Y."),
            f"Risk level: {peak_row['heuristic_risk_level']}",
        )
    with g2:
        metric_card(
            "Peak score",
            f"{peak_row['heuristic_risk_score']:.1f}",
            f"ML label: {peak_row['ml_predicted_label']}",
        )
    with g3:
        metric_card(
            "ML confidence on peak",
            f"{peak_row['ml_prediction_confidence']:.2f}",
            f"Trend delta (7d): {trend_delta:+.1f}",
        )

    fig = build_timeline_figure(active_df, selected_city)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sector recommendations")
    rec1, rec2, rec3 = st.columns(3)
    with rec1:
        render_list_card("Preporuke za grad", actions["city"])
    with rec2:
        render_list_card("Preporuke za javne službe", actions["services"])
    with rec3:
        render_list_card("Preporuke za turizam", actions["tourism"])

    st.markdown("### Alert escalation logic")
    esc1, esc2, esc3 = st.columns(3)
    with esc1:
        render_list_card("Što napraviti odmah", escalation["immediately"])
    with esc2:
        render_list_card("Što napraviti u 24h", escalation["within_24h"])
    with esc3:
        render_list_card("Što napraviti u 72h", escalation["within_72h"])

    st.markdown("### Operativni pregled po danima")
    timeline_df = active_df[
        [
            "date",
            "ml_predicted_label",
            "ml_prediction_confidence",
            "heuristic_risk_level",
            "heuristic_risk_score",
            "temp_max",
            "apparent_temp_max",
        ]
    ].copy()
    day_table = to_display_date(timeline_df)
    st.dataframe(day_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Report export")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇ Download executive brief (.txt)",
            data=executive_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_executive_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_exec_action_{selected_city}",
        )
    with d2:
        st.download_button(
            "⬇ Download operational table (.csv)",
            data=csv_bytes(day_table),
            file_name=f"heatsafe_hr_operational_table_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_operational_csv_{selected_city}",
        )
    with d3:
        st.download_button(
            "⬇ Download scenario brief (.txt)",
            data=scenario_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_scenario_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_scenario_action_{selected_city}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("### Event / Tourism Risk Check")

    day_options = active_df["date"].dt.strftime("%d.%m.%Y.").tolist()
    date_map = dict(zip(day_options, active_df["date"]))

    event_name = st.text_input("Naziv događaja", value="Summer Evening Event")

    e1, e2 = st.columns(2)
    with e1:
        selected_day_str = st.selectbox("Datum događaja", day_options)
    with e2:
        time_slot = st.selectbox(
            "Vrijeme događaja",
            ["Morning", "Midday / Afternoon", "Evening"],
            index=1,
        )

    e3, e4 = st.columns(2)
    with e3:
        event_type = st.selectbox(
            "Tip događaja",
            [
                "Outdoor concert",
                "Sports event",
                "Walking tour",
                "Festival / fair",
                "Beach / outdoor leisure",
                "City event / ceremony",
                "General tourism activity",
            ],
            index=2,
        )
    with e4:
        attendees = st.number_input("Broj sudionika / gostiju", min_value=1, max_value=50000, value=250, step=50)

    vulnerable_groups = st.toggle("Uključene osjetljive skupine", value=True)

    selected_date = date_map[selected_day_str]
    selected_row = active_df[active_df["date"] == selected_date].iloc[0]

    event_assessment = assess_event_risk(
        selected_row,
        event_type=event_type,
        attendees=int(attendees),
        time_slot=time_slot,
        vulnerable_groups=vulnerable_groups,
    )

    event_brief = build_event_brief(
        city_name=selected_city,
        event_name=event_name,
        event_date_str=selected_day_str,
        event_type=event_type,
        attendees=int(attendees),
        time_slot=time_slot,
        vulnerable_groups=vulnerable_groups,
        forecast_row=selected_row,
        assessment=event_assessment,
    )

    l1, l2 = st.columns([1, 1])

    with l1:
        st.markdown("### Rezultat procjene događaja")
        x1, x2, x3 = st.columns(3)
        with x1:
            metric_card("Event risk score", f"{event_assessment.event_score:.1f}")
        with x2:
            metric_card("Event risk level", event_assessment.event_level)
        with x3:
            metric_card("Readiness status", event_assessment.readiness_status)
        badge(event_assessment.recommendation, readiness_to_color(event_assessment.readiness_status))

    with l2:
        render_list_card("Action items", event_assessment.action_items)

    st.markdown("### Signal iza procjene")
    signal_df = pd.DataFrame(
        [
            {"Metric": "Active risk level", "Value": selected_row["heuristic_risk_level"]},
            {"Metric": "Active risk score", "Value": f"{selected_row['heuristic_risk_score']:.1f}"},
            {"Metric": "ML label", "Value": selected_row["ml_predicted_label"]},
            {"Metric": "ML confidence", "Value": f"{selected_row['ml_prediction_confidence']:.2f}"},
            {"Metric": "Apparent temp max", "Value": f"{selected_row['apparent_temp_max']:.1f} °C"},
            {"Metric": "Max temperatura", "Value": f"{selected_row['temp_max']:.1f} °C"},
        ]
    )
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Event export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download event brief (.txt)",
            data=event_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_event_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_event_brief_{selected_city}_{selected_day_str}_{event_type}",
        )
    with c2:
        event_table = pd.DataFrame(
            [
                {
                    "event_name": event_name,
                    "city": selected_city,
                    "date": selected_day_str,
                    "event_type": event_type,
                    "attendees": int(attendees),
                    "time_slot": time_slot,
                    "vulnerable_groups": vulnerable_groups,
                    "event_score": event_assessment.event_score,
                    "event_level": event_assessment.event_level,
                    "readiness_status": event_assessment.readiness_status,
                    "recommendation": event_assessment.recommendation,
                }
            ]
        )
        st.download_button(
            "⬇ Download event summary (.csv)",
            data=csv_bytes(event_table),
            file_name=f"heatsafe_hr_event_summary_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_event_csv_{selected_city}_{selected_day_str}_{event_type}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("### Executive Summary")

    st.markdown("#### Executive brief")
    st.code(executive_brief, language="text")

    st.markdown("#### Scenario comparison brief")
    st.code(scenario_brief, language="text")

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("### Brief export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download executive brief (.txt)",
            data=executive_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_executive_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_exec_summary_{selected_city}",
        )
    with c2:
        st.download_button(
            "⬇ Download scenario brief (.txt)",
            data=scenario_brief.encode("utf-8"),
            file_name=f"heatsafe_hr_scenario_brief_{selected_city}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_scenario_summary_{selected_city}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    ex1, ex2 = st.columns(2)

    with ex1:
        render_text_card(
            "Peak executive signal",
            f"""
            <div style="margin-bottom:0.6rem;">{f'<span class="status-pill" style="background:{risk_to_color(summary["next_7d_peak_level"])};">{summary["next_7d_peak_level"]}</span>'}</div>
            <b>Peak date:</b> {summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}<br>
            <b>Peak score:</b> {summary['next_7d_peak_score']:.1f}<br>
            <b>Next 24h ML confidence:</b> {summary['next_24h_confidence']:.2f}
            """,
        )

    with ex2:
        render_text_card(
            "Readiness summary",
            f"""
            <div style="margin-bottom:0.6rem;">{f'<span class="status-pill" style="background:{readiness_to_color(summary["readiness_status"])};">{summary["readiness_status"]}</span>'}</div>
            <b>Next 24h risk:</b> {summary['next_24h_level']}<br>
            <b>Next 72h peak:</b> {summary['next_72h_peak_level']}<br>
            <b>High-risk days (7d):</b> {summary['high_risk_days']}
            """,
        )

    st.success(
        """
        HeatSafe HR ovdje radi kao alat za odlučivanje:
        ne prikazuje samo prognozu, nego daje status pripravnosti, preporuke i operativni brief
        za grad, javne službe i turistički sektor.
        """
    )