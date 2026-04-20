from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def impact_band_from_peak(peak_level: str, escalation_label: str | None = None) -> str:
    if peak_level == "Vrlo visok":
        return "Severe operational impact"
    if peak_level == "Visok":
        return "High operational impact"
    if peak_level == "Umjeren" and escalation_label == "Likely escalation":
        return "High operational impact"
    if peak_level == "Umjeren":
        return "Moderate operational impact"
    if escalation_label == "Likely escalation":
        return "Moderate operational impact"
    return "Low operational impact"


def identify_primary_impacts(summary: dict[str, Any], escalation_label: str | None = None) -> list[str]:
    impacts: list[str] = []

    high_risk_days = int(summary.get("high_risk_days", 0))
    next24 = str(summary.get("next_24h_level", "Nizak"))
    peak7d = str(summary.get("next_7d_peak_level", "Nizak"))

    if next24 in ["Visok", "Vrlo visok"]:
        impacts.append("Immediate pressure on public-health and heat-safety communication")
    if peak7d in ["Visok", "Vrlo visok"]:
        impacts.append("Elevated readiness needed for city operations and emergency coordination")
    if high_risk_days >= 2:
        impacts.append("Sustained multi-day heat exposure risk for elderly people, workers, and tourists")
    if escalation_label == "Likely escalation":
        impacts.append("Early-warning signal indicates likely worsening within 72 hours")
    if not impacts:
        impacts.append("No major operational disruption expected, continue monitoring")

    return impacts


def identify_priority_groups(summary: dict[str, Any], escalation_label: str | None = None) -> list[str]:
    groups = [
        "Older adults and chronically ill residents",
        "Children and families in exposed urban areas",
        "Outdoor workers",
        "Tourists and event visitors",
    ]

    if summary.get("next_7d_peak_level") in ["Visok", "Vrlo visok"] or escalation_label == "Likely escalation":
        groups.append("City services and emergency coordination teams")

    return groups


def build_operational_triggers(summary: dict[str, Any], escalation_label: str | None = None) -> list[str]:
    triggers: list[str] = []

    if summary.get("next_24h_level") in ["Umjeren", "Visok", "Vrlo visok"]:
        triggers.append("Issue preventive public communication within 24 hours")

    if summary.get("next_7d_peak_level") in ["Visok", "Vrlo visok"]:
        triggers.append("Prepare cooling, hydration, and public-support measures")

    if int(summary.get("high_risk_days", 0)) >= 2:
        triggers.append("Monitor cumulative heat exposure and staff/service fatigue")

    if escalation_label == "Likely escalation":
        triggers.append("Activate early-warning posture before peak conditions arrive")

    if not triggers:
        triggers.append("Maintain baseline monitoring and review next forecast update")

    return triggers


def build_civil_protection_executive_brief(
    city: str,
    summary: dict[str, Any],
    escalation_probability: float | None = None,
    escalation_label: str | None = None,
    recommended_resources: pd.DataFrame | None = None,
    scenario_enabled: bool = False,
    temperature_delta: float = 0.0,
    humidity_delta: float = 0.0,
    wind_delta: float = 0.0,
) -> str:
    impact_band = impact_band_from_peak(summary["next_7d_peak_level"], escalation_label)
    primary_impacts = identify_primary_impacts(summary, escalation_label)
    priority_groups = identify_priority_groups(summary, escalation_label)
    triggers = build_operational_triggers(summary, escalation_label)

    scenario_line = "Scenario mode disabled"
    if scenario_enabled:
        scenario_line = (
            f"Scenario mode enabled | ΔT {temperature_delta:+.1f} °C | "
            f"ΔRH {humidity_delta:+.1f}% | ΔWind {wind_delta:+.1f} m/s"
        )

    peak_date = pd.to_datetime(summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")

    escalation_block = "V3 escalation signal not available"
    if escalation_probability is not None and escalation_label is not None:
        escalation_block = (
            f"72h escalation probability: {escalation_probability:.2f} | "
            f"Signal: {escalation_label}"
        )

    resources_block = "No recommended resource points available."
    if recommended_resources is not None and not recommended_resources.empty:
        lines = []
        for _, row in recommended_resources.head(3).iterrows():
            lines.append(
                f"- {row.get('resource_name', 'Unknown')} | {row.get('address', 'N/A')} | "
                f"Verified: {row.get('verified_status', 'N/A')} | "
                f"Indoor cooling: {row.get('indoor_cooling', 'N/A')} | "
                f"Water: {row.get('water_available', 'N/A')}"
            )
        resources_block = "\n".join(lines)

    impacts_block = "\n".join(f"- {item}" for item in primary_impacts)
    groups_block = "\n".join(f"- {item}" for item in priority_groups)
    triggers_block = "\n".join(f"- {item}" for item in triggers)

    return f'''
HEATSAFE HR — EXECUTIVE SUMMARY FOR CIVIL PROTECTION

Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
City: {city}
{scenario_line}

SITUATIONAL OVERVIEW
- Impact band: {impact_band}
- Readiness status: {summary['readiness_status']}
- Next 24h risk: {summary['next_24h_level']} ({summary['next_24h_score']:.1f})
- Next 72h peak: {summary['next_72h_peak_level']} ({summary['next_72h_peak_score']:.1f})
- Next 7d peak: {summary['next_7d_peak_level']} ({summary['next_7d_peak_score']:.1f})
- Peak date: {peak_date}
- High-risk days (7d): {summary['high_risk_days']}
- {escalation_block}

PRIMARY IMPACTS
{impacts_block}

PRIORITY GROUPS
{groups_block}

OPERATIONAL TRIGGERS
{triggers_block}

RECOMMENDED RESOURCE POINTS
{resources_block}
'''.strip()