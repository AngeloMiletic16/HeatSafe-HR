from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ALERT_COLOR_MAP = {
    "Monitoring Notice": "#64748b",
    "Heat Advisory": "#e6a700",
    "Heat Warning": "#e67e22",
    "Critical Alert": "#c0392b",
}

ALERT_SEVERITY_ORDER = {
    "Monitoring Notice": 1,
    "Heat Advisory": 2,
    "Heat Warning": 3,
    "Critical Alert": 4,
}


def get_alert_level(
    summary: dict[str, Any],
    escalation_probability: float | None = None,
    escalation_label: str | None = None,
) -> dict[str, Any]:
    next24 = str(summary["next_24h_level"])
    peak7d = str(summary["next_7d_peak_level"])
    high_risk_days = int(summary["high_risk_days"])
    readiness = str(summary["readiness_status"])
    next24_score = float(summary["next_24h_score"])
    peak7d_score = float(summary["next_7d_peak_score"])

    if (
        next24 == "Vrlo visok"
        or peak7d == "Vrlo visok"
        or readiness == "Critical Preparedness"
        or peak7d_score >= 75
    ):
        alert = {
            "alert_severity": "Critical Alert",
            "alert_issued": True,
            "target_audience": [
                "City leadership",
                "Emergency services",
                "Public health",
                "Tourism sector",
                "Citizens",
            ],
            "operator_summary": (
                "Very high heat-risk signal detected. Immediate coordination and public-facing action required."
            ),
            "immediate_actions": [
                "Activate critical preparedness communication.",
                "Issue public-facing alert for vulnerable groups and visitors.",
                "Prioritize cooling, hydration, and health-support readiness.",
            ],
        }
    elif (
        next24 == "Visok"
        or peak7d == "Visok"
        or high_risk_days >= 2
        or next24_score >= 50
        or peak7d_score >= 50
    ):
        alert = {
            "alert_severity": "Heat Warning",
            "alert_issued": True,
            "target_audience": [
                "City operations",
                "Public services",
                "Event organizers",
                "Tourism operators",
                "Citizens",
            ],
            "operator_summary": (
                "High heat-risk signal detected. Operational readiness should be elevated and preventive messaging issued."
            ),
            "immediate_actions": [
                "Escalate readiness to operational teams.",
                "Prepare public guidance and event/tourism precautions.",
                "Monitor next 24h and next 72h developments closely.",
            ],
        }
    elif (
        next24 == "Umjeren"
        or peak7d == "Umjeren"
        or high_risk_days >= 1
        or next24_score >= 25
        or peak7d_score >= 25
    ):
        alert = {
            "alert_severity": "Heat Advisory",
            "alert_issued": True,
            "target_audience": [
                "City services",
                "Tourism stakeholders",
                "Vulnerable citizens",
            ],
            "operator_summary": (
                "Moderate heat-risk signal detected. Preventive communication and monitoring actions are recommended."
            ),
            "immediate_actions": [
                "Prepare preventive guidance for the public.",
                "Inform tourism and event stakeholders of elevated heat conditions.",
                "Track trend changes and update readiness if forecast worsens.",
            ],
        }
    else:
        alert = {
            "alert_severity": "Monitoring Notice",
            "alert_issued": False,
            "target_audience": [
                "Internal monitoring",
            ],
            "operator_summary": (
                "No immediate alert issuance required. Continue monitoring and maintain baseline readiness."
            ),
            "immediate_actions": [
                "Continue baseline monitoring.",
                "Review updated forecast if conditions change.",
                "No public escalation required at this stage.",
            ],
        }

    # ---- V3 escalation overlay ----
    if escalation_probability is not None and escalation_label is not None:
        base_order = ALERT_SEVERITY_ORDER[alert["alert_severity"]]

        # Ako v3 vidi jaku eskalaciju, dižemo minimalnu severity razinu
        if escalation_probability >= 0.80 and base_order < ALERT_SEVERITY_ORDER["Heat Warning"]:
            alert["alert_severity"] = "Heat Warning"
            alert["alert_issued"] = True

        elif escalation_probability >= 0.60 and base_order < ALERT_SEVERITY_ORDER["Heat Advisory"]:
            alert["alert_severity"] = "Heat Advisory"
            alert["alert_issued"] = True

        if escalation_probability >= 0.60:
            extra_audience = [
                "City operations",
                "Public services",
            ]
            merged_audience = list(dict.fromkeys(alert["target_audience"] + extra_audience))
            alert["target_audience"] = merged_audience

            v3_line = (
                f" V3 early-warning signal indicates {escalation_label.lower()} "
                f"(72h escalation probability = {escalation_probability:.2f})."
            )
            alert["operator_summary"] = alert["operator_summary"] + v3_line

            early_action = "Review V3 escalation signal and prepare early-warning posture."
            alert["immediate_actions"] = [early_action] + alert["immediate_actions"]

    return alert


def build_alert_package(
    city: str,
    summary: dict[str, Any],
    alert: dict[str, Any],
    scenario_enabled: bool,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
    escalation_probability: float | None = None,
    escalation_label: str | None = None,
) -> str:
    peak_date = pd.to_datetime(summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")
    scenario_line = "Scenario mode enabled" if scenario_enabled else "Scenario mode disabled"
    if scenario_enabled:
        scenario_line += (
            f" | ΔT {temperature_delta:+.1f} °C"
            f" | ΔRH {humidity_delta:+.1f}%"
            f" | ΔWind {wind_delta:+.1f} m/s"
        )

    target_audience = ", ".join(alert["target_audience"])
    actions_block = "\n".join(f"- {item}" for item in alert["immediate_actions"])

    escalation_block = ""
    if escalation_probability is not None and escalation_label is not None:
        escalation_block = (
            f"- V3 escalation probability (72h): {escalation_probability:.2f}\n"
            f"- V3 escalation signal: {escalation_label}\n"
        )

    return f"""
HEATSAFE HR — ALERT PACKAGE

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
City: {city}
{scenario_line}

ALERT STATUS
- Alert severity: {alert['alert_severity']}
- Alert issued: {"Yes" if alert["alert_issued"] else "No"}
- Target audience: {target_audience}
{escalation_block}
RISK SUMMARY
- Next 24h risk: {summary['next_24h_level']} ({summary['next_24h_score']:.1f})
- Next 24h ML label: {summary['next_24h_ml_label']}
- Next 24h ML confidence: {summary['next_24h_confidence']:.2f}
- Next 72h peak: {summary['next_72h_peak_level']} ({summary['next_72h_peak_score']:.1f})
- Next 7d peak: {summary['next_7d_peak_level']} ({summary['next_7d_peak_score']:.1f})
- Peak date: {peak_date}
- High-risk days: {summary['high_risk_days']}
- Readiness status: {summary['readiness_status']}

OPERATOR SUMMARY
{alert['operator_summary']}

IMMEDIATE ACTIONS
{actions_block}
""".strip()


def build_operator_row(
    city: str,
    summary: dict[str, Any],
    alert: dict[str, Any],
    scenario_enabled: bool,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
    escalation_probability: float | None = None,
    escalation_label: str | None = None,
) -> dict[str, Any]:
    return {
        "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "city": city,
        "readiness": summary["readiness_status"],
        "next_24h_risk": summary["next_24h_level"],
        "next_24h_score": round(float(summary["next_24h_score"]), 1),
        "peak": summary["next_7d_peak_level"],
        "peak_score": round(float(summary["next_7d_peak_score"]), 1),
        "peak_date": pd.to_datetime(summary["next_7d_peak_date"]).strftime("%Y-%m-%d"),
        "high_risk_days": int(summary["high_risk_days"]),
        "alert_severity": alert["alert_severity"],
        "alert_issued": "Yes" if alert["alert_issued"] else "No",
        "target_audience": ", ".join(alert["target_audience"]),
        "operator_summary": alert["operator_summary"],
        "scenario_enabled": "Yes" if scenario_enabled else "No",
        "temperature_delta": temperature_delta,
        "humidity_delta": humidity_delta,
        "wind_delta": wind_delta,
        "escalation_probability_72h": round(float(escalation_probability), 4) if escalation_probability is not None else None,
        "escalation_label_72h": escalation_label,
    }


def append_alert_history(snapshot_df: pd.DataFrame, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, snapshot_df], ignore_index=True)
    else:
        combined = snapshot_df.copy()

    combined.to_csv(history_path, index=False)


def load_alert_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        return pd.DataFrame()

    return pd.read_csv(history_path)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")