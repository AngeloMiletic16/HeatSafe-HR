from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_public_advisory_hr(alert_row: dict[str, Any]) -> str:
    city = str(alert_row.get("city", "Grad"))
    severity = str(alert_row.get("alert_severity", "Monitoring Notice"))
    readiness = str(alert_row.get("readiness_status", "Monitoring"))
    peak_level = str(alert_row.get("next_7d_peak_level", "Nizak"))
    peak_score = _safe_float(alert_row.get("next_7d_peak_score"))
    high_risk_days = _safe_int(alert_row.get("high_risk_days"))
    escalation_label = str(alert_row.get("escalation_label_72h", "Stable"))

    if severity == "Critical Alert":
        action = (
            "Izbjegavajte boravak na otvorenom u najtoplijem dijelu dana, "
            "pratite službene upute i posebno provjeravajte starije osobe, djecu i kronične bolesnike."
        )
    elif severity == "Heat Warning":
        action = (
            "Smanjite kretanje i fizički napor na otvorenom, redovito pijte vodu, "
            "koristite hlad i klimatizirane prostore te pratite daljnje obavijesti."
        )
    elif severity == "Heat Advisory":
        action = (
            "Pojačajte oprez tijekom toplijeg dijela dana, prilagodite aktivnosti, "
            "nosite vodu i pratite moguće promjene uvjeta."
        )
    else:
        action = (
            "Nastavite pratiti vremenske i gradske obavijesti te primjenjujte osnovne mjere zaštite od vrućine."
        )

    return (
        f"HEATSAFE HR — JAVNA OBAVIJEST\n\n"
        f"Grad: {city}\n"
        f"Razina upozorenja: {severity}\n"
        f"Status pripravnosti: {readiness}\n"
        f"Peak rizik u sljedećih 7 dana: {peak_level} ({peak_score:.1f})\n"
        f"Broj high-risk dana: {high_risk_days}\n"
        f"72h escalation signal: {escalation_label}\n\n"
        f"Preporuka za građane:\n"
        f"{action}"
    )


def build_tourist_advisory_en(alert_row: dict[str, Any]) -> str:
    city = str(alert_row.get("city", "City"))
    severity = str(alert_row.get("alert_severity", "Monitoring Notice"))
    peak_level = str(alert_row.get("next_7d_peak_level", "Low"))
    peak_score = _safe_float(alert_row.get("next_7d_peak_score"))
    escalation_label = str(alert_row.get("escalation_label_72h", "Stable"))

    if severity == "Critical Alert":
        action = (
            "Avoid outdoor activity during the hottest hours, reduce walking exposure, "
            "use shaded or air-conditioned spaces, and follow local safety guidance."
        )
    elif severity == "Heat Warning":
        action = (
            "Plan sightseeing early or late in the day, drink water frequently, "
            "avoid long queues and intense physical activity, and use cooling spaces when possible."
        )
    elif severity == "Heat Advisory":
        action = (
            "Use extra caution during warmer hours, carry water, wear light clothing, "
            "and adjust walking tours or outdoor plans if needed."
        )
    else:
        action = (
            "Continue normal activity with standard heat precautions such as hydration, shade, and sun protection."
        )

    return (
        f"HEATSAFE HR — TOURIST HEAT ADVISORY\n\n"
        f"City: {city}\n"
        f"Alert severity: {severity}\n"
        f"Projected 7-day peak: {peak_level} ({peak_score:.1f})\n"
        f"72h escalation signal: {escalation_label}\n\n"
        f"Recommended visitor guidance:\n"
        f"{action}"
    )


def build_media_brief(alert_row: dict[str, Any]) -> str:
    city = str(alert_row.get("city", "Grad"))
    severity = str(alert_row.get("alert_severity", "Monitoring Notice"))
    readiness = str(alert_row.get("readiness_status", "Monitoring"))
    peak_level = str(alert_row.get("next_7d_peak_level", "Nizak"))
    peak_score = _safe_float(alert_row.get("next_7d_peak_score"))
    escalation_probability = _safe_float(alert_row.get("escalation_probability_72h"))
    escalation_label = str(alert_row.get("escalation_label_72h", "Stable"))

    return (
        f"HEATSAFE HR — MEDIA BRIEF\n\n"
        f"Za grad {city} sustav trenutno izdaje razinu upozorenja '{severity}' "
        f"uz readiness status '{readiness}'. "
        f"U sljedećih 7 dana očekuje se peak razina rizika '{peak_level}' "
        f"sa scoreom {peak_score:.1f}. "
        f"V3 early-warning model daje 72h escalation probability {escalation_probability:.2f} "
        f"uz signal '{escalation_label}'. "
        f"Preporučuje se pravodobno informiranje građana, turista i ranjivih skupina."
    )


def build_operator_sms(alert_row: dict[str, Any]) -> str:
    city = str(alert_row.get("city", "Grad"))
    severity = str(alert_row.get("alert_severity", "Monitoring Notice"))
    readiness = str(alert_row.get("readiness_status", "Monitoring"))
    peak_score = _safe_float(alert_row.get("next_7d_peak_score"))
    escalation_probability = _safe_float(alert_row.get("escalation_probability_72h"))
    escalation_label = str(alert_row.get("escalation_label_72h", "Stable"))

    return (
        f"{city} | {severity} | Readiness: {readiness} | "
        f"Peak7d: {peak_score:.1f} | Esc72h: {escalation_label} ({escalation_probability:.2f})"
    )


def build_social_post_hr(alert_row: dict[str, Any]) -> str:
    city = str(alert_row.get("city", "Grad"))
    severity = str(alert_row.get("alert_severity", "Monitoring Notice"))

    if severity in ["Heat Warning", "Critical Alert"]:
        action = "Ograničite boravak na suncu, pijte vodu i provjerite starije i ranjive osobe."
    elif severity == "Heat Advisory":
        action = "Prilagodite aktivnosti, nosite vodu i pratite daljnje obavijesti."
    else:
        action = "Pratite uvjete i primjenjujte osnovne mjere zaštite od vrućine."

    return f"HeatSafe HR | {city} | {severity}. {action}"


def build_social_post_en(alert_row: dict[str, Any]) -> str:
    city = str(alert_row.get("city", "City"))
    severity = str(alert_row.get("alert_severity", "Monitoring Notice"))

    if severity in ["Heat Warning", "Critical Alert"]:
        action = "Limit sun exposure, drink water, and check on vulnerable people."
    elif severity == "Heat Advisory":
        action = "Adjust activities, carry water, and follow further guidance."
    else:
        action = "Monitor conditions and use standard heat precautions."

    return f"HeatSafe HR | {city} | {severity}. {action}"


def build_alert_communication_package(alert_row: dict[str, Any]) -> dict[str, str]:
    return {
        "public_advisory_hr": build_public_advisory_hr(alert_row),
        "tourist_advisory_en": build_tourist_advisory_en(alert_row),
        "media_brief": build_media_brief(alert_row),
        "operator_sms": build_operator_sms(alert_row),
        "social_post_hr": build_social_post_hr(alert_row),
        "social_post_en": build_social_post_en(alert_row),
    }