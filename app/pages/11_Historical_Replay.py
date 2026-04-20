from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.alert_engine import csv_bytes, get_alert_level
from src.config import DEFAULT_CITY
from src.decision_engine import build_escalation_plan, build_sector_actions
from src.escalation_engine import get_city_escalation_summary_by_date

st.set_page_config(page_title="Historical Replay", page_icon="⏪", layout="wide")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_daily_with_risk.csv"

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}

ALERT_COLOR_MAP = {
    "Monitoring Notice": "#64748b",
    "Heat Advisory": "#e6a700",
    "Heat Warning": "#e67e22",
    "Critical Alert": "#c0392b",
}

ALERT_NUMERIC_MAP = {
    "Monitoring Notice": 1,
    "Heat Advisory": 2,
    "Heat Warning": 3,
    "Critical Alert": 4,
}

READINESS_MAP = {
    "Nizak": "Monitoring",
    "Umjeren": "Prepared",
    "Visok": "Elevated Readiness",
    "Vrlo visok": "Critical Preparedness",
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

    .note-box {
        background: #eff6ff;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        color: #0f172a;
        margin: 0.7rem 0 1rem 0;
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


def pill(text: str, color: str) -> str:
    return f'<span class="status-pill" style="background:{color};">{text}</span>'


@st.cache_data
def load_daily_risk_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}. Run risk pipeline first."
        )

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)


def build_replay_summary(window_df: pd.DataFrame, city: str) -> dict:
    window_df = window_df.sort_values("date").reset_index(drop=True)

    next24 = window_df.iloc[0]
    next72 = window_df.head(3).sort_values(
        ["heat_risk_score", "apparent_temp_max"],
        ascending=[False, False],
    ).iloc[0]
    next7 = window_df.sort_values(
        ["heat_risk_score", "apparent_temp_max"],
        ascending=[False, False],
    ).iloc[0]

    high_risk_days = int(window_df["risk_level"].isin(["Visok", "Vrlo visok"]).sum())

    return {
        "city": city,
        "next_24h_level": str(next24["risk_level"]),
        "next_24h_score": float(next24["heat_risk_score"]),
        "next_24h_ml_label": str(next24["risk_level"]),
        "next_24h_confidence": 1.0,
        "next_72h_peak_level": str(next72["risk_level"]),
        "next_72h_peak_score": float(next72["heat_risk_score"]),
        "next_7d_peak_level": str(next7["risk_level"]),
        "next_7d_peak_score": float(next7["heat_risk_score"]),
        "next_7d_peak_date": pd.to_datetime(next7["date"]),
        "high_risk_days": high_risk_days,
        "readiness_status": READINESS_MAP.get(str(next7["risk_level"]), "Monitoring"),
    }


def build_replay_log(city_df: pd.DataFrame, issue_period_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, issue_row in issue_period_df.iterrows():
        issue_date = pd.to_datetime(issue_row["date"])
        end_date = issue_date + pd.Timedelta(days=6)

        window_df = city_df[
            (city_df["date"] >= issue_date) & (city_df["date"] <= end_date)
        ].copy()

        if window_df.empty:
            continue

        summary = build_replay_summary(window_df, str(issue_row["city"]))

        escalation_summary = get_city_escalation_summary_by_date(
            str(issue_row["city"]),
            issue_date,
        )

        alert = get_alert_level(
            summary,
            escalation_probability=escalation_summary["escalation_probability_72h"],
            escalation_label=escalation_summary["escalation_label_72h"],
        )

        rows.append(
            {
                "issue_date": issue_date,
                "city": str(issue_row["city"]),
                "next_24h_level": summary["next_24h_level"],
                "next_24h_score": round(summary["next_24h_score"], 1),
                "next_72h_peak_level": summary["next_72h_peak_level"],
                "next_72h_peak_score": round(summary["next_72h_peak_score"], 1),
                "next_7d_peak_level": summary["next_7d_peak_level"],
                "next_7d_peak_score": round(summary["next_7d_peak_score"], 1),
                "next_7d_peak_date": summary["next_7d_peak_date"],
                "high_risk_days": summary["high_risk_days"],
                "readiness_status": summary["readiness_status"],
                "alert_severity": alert["alert_severity"],
                "alert_issued": "Yes" if alert["alert_issued"] else "No",
                "target_audience": ", ".join(alert["target_audience"]),
                "operator_summary": alert["operator_summary"],
                "actions_now": " | ".join(alert["immediate_actions"]),
                "escalation_probability_72h": round(float(escalation_summary["escalation_probability_72h"]), 4),
                "escalation_label_72h": escalation_summary["escalation_label_72h"],
                "escalation_operator_message": escalation_summary["operator_message"],
            }
        )

    replay_df = pd.DataFrame(rows)
    if not replay_df.empty:
        replay_df = replay_df.sort_values("issue_date").reset_index(drop=True)
        replay_df["alert_severity_numeric"] = replay_df["alert_severity"].map(ALERT_NUMERIC_MAP)
    return replay_df


st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">⏪ Historical Replay / Case Study</div>
        <div class="page-hero-subtitle">
            Replay modul pokazuje što bi HeatSafe HR napravio u odabranom povijesnom periodu.
            Time projekt dobiva evaluation layer: ne samo “što radi danas”, nego i “što bi preporučio
            u stvarnoj toplinskoj epizodi”.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_daily_risk_data()
except Exception as exc:
    st.error(f"Historical replay nije dostupan: {exc}")
    st.stop()

default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = CITIES.index(default_city) if default_city in CITIES else 0

top1, top2 = st.columns([1.1, 1.3])
with top1:
    selected_city = st.selectbox("Odaberi grad", CITIES, index=default_index)
    st.session_state.selected_city = selected_city

city_df = df[df["city"] == selected_city].copy().sort_values("date")

min_date = city_df["date"].min().date()
max_date = city_df["date"].max().date()

default_end = max_date
default_start = max(min_date, default_end - pd.Timedelta(days=13))

with top2:
    selected_range = st.date_input(
        "Odaberi period za replay",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
    )

if not isinstance(selected_range, (list, tuple)) or len(selected_range) != 2:
    st.warning("Odaberi početni i završni datum.")
    st.stop()

start_date = pd.to_datetime(selected_range[0])
end_date = pd.to_datetime(selected_range[1])

issue_period_df = city_df[
    (city_df["date"] >= start_date) & (city_df["date"] <= end_date)
].copy()

if issue_period_df.empty:
    st.warning("Nema podataka za odabrani period.")
    st.stop()

replay_df = build_replay_log(city_df, issue_period_df)

if replay_df.empty:
    st.warning("Replay log nije moguće izgraditi za odabrani period.")
    st.stop()

critical_count = int((replay_df["alert_severity"] == "Critical Alert").sum())
warning_count = int(replay_df["alert_severity"].isin(["Heat Warning", "Critical Alert"]).sum())
first_alert_row = replay_df[replay_df["alert_issued"] == "Yes"].head(1)
first_alert_date = (
    pd.to_datetime(first_alert_row.iloc[0]["issue_date"]).strftime("%d.%m.%Y.")
    if not first_alert_row.empty
    else "No alert"
)

peak_row = replay_df.sort_values(
    ["next_7d_peak_score", "high_risk_days"],
    ascending=[False, False],
).iloc[0]

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Replay days", str(len(replay_df)), selected_city)
with m2:
    metric_card("First alert", str(first_alert_date), "First issued signal")
with m3:
    metric_card("Warning or higher", str(warning_count), "Heat Warning + Critical")
with m4:
    metric_card("Critical alerts", str(critical_count), "Highest severity")

st.markdown(
    """
    <div class="note-box">
        <b>Kako čitati replay:</b> svaki red i svaki issue date predstavljaju dan na koji je sustav “pokrenut”.
        Zatim se gleda što se stvarno dogodilo kroz sljedećih 7 dana i iz toga se računa
        what-would-the-system-do snapshot.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Replay timeline", "Case study day", "Replay log table"])

with tabs[0]:
    st.markdown("### Historical risk in selected period")

    history_fig = px.bar(
        issue_period_df,
        x="date",
        y="heat_risk_score",
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        title=f"Observed historical heat risk — {selected_city}",
        hover_data={
            "temp_max": True,
            "temp_min": True,
            "apparent_temp_max": True,
            "humidity_mean": True,
            "wind_speed_mean": True,
            "date": False,
        },
    )
    history_fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Datum",
        yaxis_title="Heat Risk Score",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    history_fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(history_fig, use_container_width=True)

    st.markdown("### What alert would the system issue?")

    alert_fig = px.line(
        replay_df,
        x="issue_date",
        y="alert_severity_numeric",
        markers=True,
        color="alert_severity",
        color_discrete_map=ALERT_COLOR_MAP,
        title=f"Replay alert severity over time — {selected_city}",
        hover_data={
            "next_24h_level": True,
            "next_24h_score": True,
            "next_7d_peak_level": True,
            "next_7d_peak_score": True,
            "high_risk_days": True,
            "issue_date": False,
            "alert_severity_numeric": False,
        },
    )
    alert_fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Issue date",
        yaxis_title="Alert severity level",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    alert_fig.update_yaxes(
        tickvals=[1, 2, 3, 4],
        ticktext=["Monitoring", "Advisory", "Warning", "Critical"],
        gridcolor="rgba(15,23,42,0.08)",
    )
    st.plotly_chart(alert_fig, use_container_width=True)

with tabs[1]:
    st.markdown("### Case study day")

    replay_dates = replay_df["issue_date"].dt.strftime("%d.%m.%Y.").tolist()
    selected_issue_str = st.selectbox("Odaberi issue date", replay_dates, index=0)
    selected_issue_date = pd.to_datetime(selected_issue_str, dayfirst=True)

    selected_replay = replay_df[replay_df["issue_date"] == selected_issue_date].iloc[0]
    end_window = selected_issue_date + pd.Timedelta(days=6)

    selected_window_df = city_df[
        (city_df["date"] >= selected_issue_date) & (city_df["date"] <= end_window)
    ].copy()

    replay_summary = build_replay_summary(selected_window_df, selected_city)
    replay_alert = get_alert_level(replay_summary)
    replay_escalation = build_escalation_plan(replay_summary["next_7d_peak_level"])
    replay_actions = build_sector_actions(replay_summary["next_7d_peak_level"])

    badge_html = pill(
        selected_replay["alert_severity"],
        ALERT_COLOR_MAP.get(selected_replay["alert_severity"], "#64748b"),
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Issue date", selected_issue_str, selected_city)
    with c2:
        metric_card("Next 24h", selected_replay["next_24h_level"], f"{selected_replay['next_24h_score']:.1f}")
    with c3:
        metric_card("Next 7d peak", selected_replay["next_7d_peak_level"], f"{selected_replay['next_7d_peak_score']:.1f}")
    with c4:
        metric_card("Readiness", selected_replay["readiness_status"], selected_replay["alert_severity"])
    c5, c6 = st.columns(2)
    with c5:
        metric_card(
            "72h escalation probability",
            f"{selected_replay['escalation_probability_72h']:.2f}",
            "V3 early-warning",
        )
    with c6:
        metric_card(
            "Escalation signal",
            selected_replay["escalation_label_72h"],
            "Historical V3 signal",
        )

    st.info(selected_replay["escalation_operator_message"])

    left, right = st.columns(2)
    with left:
        panel(
            "Operator summary",
            f"""
            <b>Target audience:</b> {selected_replay['target_audience']}<br><br>
            {selected_replay['operator_summary']}
            """,
        )
    with right:
        panel(
            "Observed window",
            f"""
            <b>Replay window:</b> {selected_issue_date.strftime('%d.%m.%Y.')} - {end_window.strftime('%d.%m.%Y.')}<br>
            <b>Peak date:</b> {pd.to_datetime(replay_summary['next_7d_peak_date']).strftime('%d.%m.%Y.')}<br>
            <b>High-risk days:</b> {replay_summary['high_risk_days']}<br>
            <b>Alert issued:</b> {"Yes" if replay_alert["alert_issued"] else "No"}
            """,
        )

    st.markdown("### Sector actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        panel("Za grad", "<br>".join(f"• {x}" for x in replay_actions["city"]))
    with a2:
        panel("Za javne službe", "<br>".join(f"• {x}" for x in replay_actions["services"]))
    with a3:
        panel("Za turizam", "<br>".join(f"• {x}" for x in replay_actions["tourism"]))

    st.markdown("### Escalation logic")
    e1, e2, e3 = st.columns(3)
    with e1:
        panel("Što napraviti odmah", "<br>".join(f"• {x}" for x in replay_escalation["immediately"]))
    with e2:
        panel("Što napraviti u 24h", "<br>".join(f"• {x}" for x in replay_escalation["within_24h"]))
    with e3:
        panel("Što napraviti u 72h", "<br>".join(f"• {x}" for x in replay_escalation["within_72h"]))

with tabs[2]:
    st.markdown("### Replay log table")

    display_df = replay_df[
        [
            "issue_date",
            "city",
            "next_24h_level",
            "next_24h_score",
            "next_72h_peak_level",
            "next_72h_peak_score",
            "next_7d_peak_level",
            "next_7d_peak_score",
            "next_7d_peak_date",
            "high_risk_days",
            "readiness_status",
            "alert_severity",
            "alert_issued",
            "target_audience",
            "operator_summary",
            "escalation_probability_72h",
            "escalation_label_72h",
        ]
    ].copy()

    display_df["issue_date"] = pd.to_datetime(display_df["issue_date"]).dt.strftime("%d.%m.%Y.")
    display_df["next_7d_peak_date"] = pd.to_datetime(display_df["next_7d_peak_date"]).dt.strftime("%d.%m.%Y.")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download replay log (.csv)",
        data=csv_bytes(display_df),
        file_name=f"heatsafe_hr_replay_{selected_city}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"dl_replay_{selected_city}",
    )