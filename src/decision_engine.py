from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


READINESS_COLOR_MAP = {
    "Monitoring": "#2E8B57",
    "Prepared": "#E6A700",
    "Elevated Readiness": "#E67E22",
    "Critical Preparedness": "#C0392B",
}

RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}


@dataclass
class EventAssessment:
    event_score: float
    event_level: str
    readiness_status: str
    recommendation: str
    action_items: list[str]


def risk_to_readiness(risk_level: str) -> str:
    mapping = {
        "Nizak": "Monitoring",
        "Umjeren": "Prepared",
        "Visok": "Elevated Readiness",
        "Vrlo visok": "Critical Preparedness",
    }
    return mapping.get(risk_level, "Monitoring")


def risk_to_color(risk_level: str) -> str:
    return RISK_COLOR_MAP.get(risk_level, "#666666")


def readiness_to_color(readiness_status: str) -> str:
    return READINESS_COLOR_MAP.get(readiness_status, "#666666")


def get_peak_forecast_row(forecast_df: pd.DataFrame) -> pd.Series:
    return forecast_df.sort_values(
        ["heuristic_risk_score", "apparent_temp_max", "ml_prediction_confidence"],
        ascending=[False, False, False],
    ).iloc[0]


def get_next_24h_row(forecast_df: pd.DataFrame) -> pd.Series:
    return forecast_df.sort_values("date").iloc[0]


def get_next_72h_peak(forecast_df: pd.DataFrame) -> pd.Series:
    first_3 = forecast_df.sort_values("date").head(3).copy()
    return get_peak_forecast_row(first_3)


def build_sector_actions(risk_level: str) -> dict[str, list[str]]:
    if risk_level == "Nizak":
        return {
            "city": [
                "Nastaviti rutinsko praćenje vremenskih uvjeta.",
                "Održavati osnovni komunikacijski i operativni monitoring.",
                "Pripremiti pregled kapaciteta za moguće toplije dane.",
            ],
            "services": [
                "Standardni operativni režim.",
                "Povremene provjere osjetljivih skupina i domova.",
                "Praćenje opterećenja bez dodatne eskalacije.",
            ],
            "tourism": [
                "Redovito informirati goste o vremenskim uvjetima.",
                "Nisu potrebne posebne restrikcije aktivnosti.",
                "Preporučiti osnovnu hidraciju i zaštitu od sunca.",
            ],
        }

    if risk_level == "Umjeren":
        return {
            "city": [
                "Pojačati javne obavijesti o toplinskim uvjetima.",
                "Pripremiti rashladne punktove i logistiku za eskalaciju.",
                "Usuglasiti komunikaciju prema rizičnim skupinama.",
            ],
            "services": [
                "Pojačati pripravnost hitnih i javnih službi.",
                "Pojačano pratiti starije, kronične bolesnike i djecu.",
                "Pripremiti dodatne operativne smjene ako se trend pogorša.",
            ],
            "tourism": [
                "Upozoriti organizatore aktivnosti na otvorenom.",
                "Preporučiti prilagodbu termina za najtopliji dio dana.",
                "Istaknuti preporuke za vodu, hlad i zaštitu od sunca.",
            ],
        }

    if risk_level == "Visok":
        return {
            "city": [
                "Aktivirati pojačani gradski komunikacijski režim.",
                "Pripremiti rashladne točke i lokacije za pomoć građanima.",
                "Koordinirati komunalne i javne službe za pojačan odgovor.",
            ],
            "services": [
                "Povećati pripravnost hitnih, zdravstvenih i socijalnih službi.",
                "Pojačano pratiti rizične skupine i ustanove.",
                "Planirati brzo operativno reagiranje u slučaju daljnjeg pogoršanja.",
            ],
            "tourism": [
                "Prilagoditi rasporede aktivnosti na otvorenom.",
                "Preporučiti dodatne sigurnosne i zdravstvene mjere za goste.",
                "Komunicirati toplinski rizik hotelima, vodičima i organizatorima.",
            ],
        }

    return {
        "city": [
            "Aktivirati krizni režim toplinskog vala.",
            "Pokrenuti snažne javne obavijesti i preventivne mjere.",
            "Koordinirati gradske službe i prioritetno usmjeriti resurse.",
        ],
        "services": [
            "Povećati pripravnost hitnih, zdravstvenih i socijalnih sustava.",
            "Pojačati nadzor rizičnih skupina i operativnih točaka.",
            "Pripremiti odgovor na moguće povećano opterećenje sustava.",
        ],
        "tourism": [
            "Razmotriti odgodu ili snažnu prilagodbu aktivnosti na otvorenom.",
            "Obvezno informirati goste o mjerama opreza i ograničenjima.",
            "Pojačati dostupnost vode, hlada i medicinske podrške.",
        ],
    }


def build_city_readiness_summary(city_name: str, forecast_df: pd.DataFrame) -> dict[str, Any]:
    next_24h = get_next_24h_row(forecast_df)
    next_72h_peak = get_next_72h_peak(forecast_df)
    next_7d_peak = get_peak_forecast_row(forecast_df)

    high_risk_days = int(
        ((forecast_df["heuristic_risk_level"] == "Visok") | (forecast_df["heuristic_risk_level"] == "Vrlo visok")).sum()
    )

    readiness_status = risk_to_readiness(str(next_7d_peak["heuristic_risk_level"]))

    return {
        "city": city_name,
        "next_24h_level": str(next_24h["heuristic_risk_level"]),
        "next_24h_score": float(next_24h["heuristic_risk_score"]),
        "next_24h_ml_label": str(next_24h["ml_predicted_label"]),
        "next_24h_confidence": float(next_24h["ml_prediction_confidence"]),
        "next_72h_peak_level": str(next_72h_peak["heuristic_risk_level"]),
        "next_72h_peak_score": float(next_72h_peak["heuristic_risk_score"]),
        "next_7d_peak_level": str(next_7d_peak["heuristic_risk_level"]),
        "next_7d_peak_score": float(next_7d_peak["heuristic_risk_score"]),
        "next_7d_peak_date": pd.to_datetime(next_7d_peak["date"]),
        "high_risk_days": high_risk_days,
        "readiness_status": readiness_status,
    }


def assess_event_risk(
    forecast_row: pd.Series,
    event_type: str,
    attendees: int,
    time_slot: str,
    vulnerable_groups: bool,
) -> EventAssessment:
    score = float(forecast_row["heuristic_risk_score"])

    event_type_adjustments = {
        "Outdoor concert": 12,
        "Sports event": 12,
        "Walking tour": 8,
        "Festival / fair": 10,
        "Beach / outdoor leisure": 7,
        "City event / ceremony": 8,
        "General tourism activity": 6,
    }

    time_slot_adjustments = {
        "Morning": 0,
        "Midday / Afternoon": 12,
        "Evening": 4,
    }

    if attendees >= 1000:
        score += 8
    elif attendees >= 300:
        score += 5
    elif attendees >= 100:
        score += 3

    score += event_type_adjustments.get(event_type, 0)
    score += time_slot_adjustments.get(time_slot, 0)

    if vulnerable_groups:
        score += 8

    score = min(score, 100)

    if score <= 24:
        level = "Nizak"
        recommendation = "Proceed"
    elif score <= 49:
        level = "Umjeren"
        recommendation = "Proceed with precautions"
    elif score <= 74:
        level = "Visok"
        recommendation = "Modify / mitigate"
    else:
        level = "Vrlo visok"
        recommendation = "Postpone or heavily adapt"

    readiness_status = risk_to_readiness(level)

    if recommendation == "Proceed":
        action_items = [
            "Osigurati osnovnu hidraciju i komunikaciju prema sudionicima.",
            "Pratiti lokalne uvjete i eventualne promjene forecasta.",
            "Osigurati osnovne točke hlada i informiranja.",
        ]
    elif recommendation == "Proceed with precautions":
        action_items = [
            "Povećati dostupnost vode i hlada.",
            "Komunicirati preporuke sudionicima i osoblju.",
            "Pripremiti brzi odgovor za osjetljive sudionike.",
        ]
    elif recommendation == "Modify / mitigate":
        action_items = [
            "Pomaknuti termin iz najtoplijeg dijela dana ako je moguće.",
            "Povećati medicinsku i logističku spremnost.",
            "Smanjiti opterećenje sudionika i osigurati rashladne točke.",
        ]
    else:
        action_items = [
            "Razmotriti odgodu ili veliku prilagodbu događaja.",
            "Aktivirati pojačane sigurnosne i medicinske mjere.",
            "Snažno upozoriti organizatore, osoblje i sudionike.",
        ]

    return EventAssessment(
        event_score=score,
        event_level=level,
        readiness_status=readiness_status,
        recommendation=recommendation,
        action_items=action_items,
    )


def build_executive_brief(city_name: str, forecast_df: pd.DataFrame, scenario_used: bool) -> str:
    summary = build_city_readiness_summary(city_name, forecast_df)
    actions = build_sector_actions(summary["next_7d_peak_level"])

    scenario_line = "Scenario mode uključen." if scenario_used else "Scenario mode nije uključen."

    return f"""
HEATSAFE HR — EXECUTIVE BRIEF

Grad: {city_name}
Status pripravnosti: {summary['readiness_status']}
{scenario_line}

NEXT 24H
- Razina rizika: {summary['next_24h_level']}
- Score: {summary['next_24h_score']:.1f}
- ML label: {summary['next_24h_ml_label']}
- ML confidence: {summary['next_24h_confidence']:.2f}

NEXT 72H PEAK
- Razina rizika: {summary['next_72h_peak_level']}
- Score: {summary['next_72h_peak_score']:.1f}

NEXT 7D PEAK
- Datum: {summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}
- Razina rizika: {summary['next_7d_peak_level']}
- Score: {summary['next_7d_peak_score']:.1f}
- Broj visokih i vrlo visokih dana: {summary['high_risk_days']}

PREPORUKE ZA GRAD
- {actions['city'][0]}
- {actions['city'][1]}
- {actions['city'][2]}

PREPORUKE ZA JAVNE SLUŽBE
- {actions['services'][0]}
- {actions['services'][1]}
- {actions['services'][2]}

PREPORUKE ZA TURIZAM
- {actions['tourism'][0]}
- {actions['tourism'][1]}
- {actions['tourism'][2]}
""".strip()


def build_event_brief(
    city_name: str,
    event_name: str,
    event_date_str: str,
    event_type: str,
    attendees: int,
    time_slot: str,
    vulnerable_groups: bool,
    forecast_row: pd.Series,
    assessment: EventAssessment,
) -> str:
    vulnerable_text = "Da" if vulnerable_groups else "Ne"

    action_lines = "\n".join(f"- {item}" for item in assessment.action_items)

    return f"""
HEATSAFE HR — EVENT RISK BRIEF

Grad: {city_name}
Naziv događaja: {event_name}
Datum: {event_date_str}
Tip događaja: {event_type}
Vrijeme događaja: {time_slot}
Broj sudionika: {attendees}
Osjetljive skupine: {vulnerable_text}

FORECAST SIGNAL
- Baseline risk level: {forecast_row['heuristic_risk_level']}
- Baseline risk score: {forecast_row['heuristic_risk_score']:.1f}
- ML label: {forecast_row['ml_predicted_label']}
- ML confidence: {forecast_row['ml_prediction_confidence']:.2f}
- Apparent temp max: {forecast_row['apparent_temp_max']:.1f} °C
- Max temperatura: {forecast_row['temp_max']:.1f} °C

EVENT ASSESSMENT
- Event risk score: {assessment.event_score:.1f}
- Event risk level: {assessment.event_level}
- Readiness status: {assessment.readiness_status}
- Recommendation: {assessment.recommendation}

ACTION ITEMS
{action_lines}
""".strip()


def build_scenario_comparison_brief(
    city_name: str,
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> str:
    baseline_summary = build_city_readiness_summary(city_name, baseline_df)
    scenario_summary = build_city_readiness_summary(city_name, scenario_df)

    baseline_peak = get_peak_forecast_row(baseline_df)
    scenario_peak = get_peak_forecast_row(scenario_df)

    return f"""
HEATSAFE HR — SCENARIO COMPARISON BRIEF

Grad: {city_name}

SCENARIO INPUT
- Promjena temperature: {temperature_delta:+.1f} °C
- Promjena vlage: {humidity_delta:+.1f} %
- Promjena vjetra: {wind_delta:+.1f} m/s

BASELINE
- Readiness status: {baseline_summary['readiness_status']}
- Next 7d peak level: {baseline_summary['next_7d_peak_level']}
- Next 7d peak score: {baseline_summary['next_7d_peak_score']:.1f}
- Peak date: {baseline_summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}
- High-risk days: {baseline_summary['high_risk_days']}

SCENARIO
- Readiness status: {scenario_summary['readiness_status']}
- Next 7d peak level: {scenario_summary['next_7d_peak_level']}
- Next 7d peak score: {scenario_summary['next_7d_peak_score']:.1f}
- Peak date: {scenario_summary['next_7d_peak_date'].strftime('%d.%m.%Y.')}
- High-risk days: {scenario_summary['high_risk_days']}

DELTA
- Peak score delta: {scenario_summary['next_7d_peak_score'] - baseline_summary['next_7d_peak_score']:+.1f}
- High-risk day delta: {scenario_summary['high_risk_days'] - baseline_summary['high_risk_days']:+d}
- Peak apparent temperature delta: {scenario_peak['apparent_temp_max'] - baseline_peak['apparent_temp_max']:+.1f} °C
""".strip()

def build_escalation_plan(risk_level: str) -> dict[str, list[str]]:
    if risk_level == "Nizak":
        return {
            "immediately": [
                "Nastaviti rutinsko praćenje vremenskih uvjeta.",
                "Provjeriti da su komunikacijski kanali i osnovni kapaciteti spremni.",
                "Održavati osnovni monitoring rizičnih skupina.",
            ],
            "within_24h": [
                "Ažurirati internu procjenu rizika ako forecast pokaže rast temperature.",
                "Pripremiti nacrt javne obavijesti za slučaj pogoršanja.",
                "Potvrditi dostupnost osnovnih resursa za preventivni odgovor.",
            ],
            "within_72h": [
                "Pratiti razvoj forecasta i moguće ulaske u Umjereni rizik.",
                "Pregledati logistiku rashladnih punktova i terenskih kapaciteta.",
                "Zadržati readiness status na Monitoring.",
            ],
        }

    if risk_level == "Umjeren":
        return {
            "immediately": [
                "Pojačati praćenje forecasta i javnu komunikacijsku spremnost.",
                "Upozoriti ključne gradske i javne aktere na moguću eskalaciju.",
                "Pripremiti preventivne poruke za građane i posjetitelje.",
            ],
            "within_24h": [
                "Provjeriti spremnost rashladnih punktova i operativnih resursa.",
                "Pojačati koordinaciju s javnim i zdravstvenim službama.",
                "Pripremiti popis lokacija i skupina koje traže pojačani nadzor.",
            ],
            "within_72h": [
                "Ako trend raste, prijeći na Prepared ili Elevated Readiness režim.",
                "Pojačati distribuciju preporuka prema turizmu i organizatorima događaja.",
                "Pratiti broj dana koji ulaze u Visok i Vrlo visok rizik.",
            ],
        }

    if risk_level == "Visok":
        return {
            "immediately": [
                "Aktivirati pojačanu komunikaciju i operativnu koordinaciju.",
                "Pripremiti rashladne punktove i terenske resurse.",
                "Pojačati nadzor rizičnih skupina i kritičnih lokacija.",
            ],
            "within_24h": [
                "Pojačati spremnost hitnih, zdravstvenih i socijalnih službi.",
                "Objaviti ciljane preporuke za građane, turizam i događaje na otvorenom.",
                "Planirati dodatne smjene ili raspoloživost ključnih timova.",
            ],
            "within_72h": [
                "Ako se trend održi ili raste, prijeći u Critical Preparedness.",
                "Razmotriti ograničenja za vanjske aktivnosti u vršnim satima.",
                "Povećati razinu koordinacije među gradskim i javnim službama.",
            ],
        }

    return {
        "immediately": [
            "Aktivirati krizni režim toplinskog vala.",
            "Pokrenuti snažnu javnu obavijest i preventivne mjere bez odgode.",
            "Prioritetno usmjeriti resurse prema najrizičnijim skupinama i lokacijama.",
        ],
        "within_24h": [
            "Povećati pripravnost hitnih, zdravstvenih i socijalnih sustava.",
            "Pojačati logistiku vode, hlada, rashladnih prostora i medicinske podrške.",
            "Aktivno pratiti događaje, turizam i operativna opterećenja sustava.",
        ],
        "within_72h": [
            "Održavati Critical Preparedness dok god forecast signal ostaje visok.",
            "Razmotriti odgodu ili snažnu prilagodbu aktivnosti na otvorenom.",
            "Kontinuirano ažurirati leadership brief i gradske operativne preporuke.",
        ],
    }