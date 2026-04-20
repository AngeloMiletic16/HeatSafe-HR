from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_PATH = PROJECT_ROOT / "data" / "resources" / "cooling_centers.csv"


def load_resources() -> pd.DataFrame:
    if not RESOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing resource file: {RESOURCE_PATH}")

    df = pd.read_csv(RESOURCE_PATH)
    return df


def score_resource(row: pd.Series, escalation_label: str) -> float:
    score = 0.0

    if row.get("verified_status") == "Verified":
        score += 4
    elif row.get("verified_status") == "Partially verified":
        score += 2

    if str(row.get("indoor_cooling")) in ["Yes", "Yes - indoor"]:
        score += 4

    if str(row.get("water_available")) == "Yes":
        score += 2

    if str(row.get("wheelchair_access")) == "Yes":
        score += 1.5

    if str(row.get("elderly_friendly")) == "Yes":
        score += 1.5

    if str(row.get("child_friendly")) == "Yes":
        score += 1.0

    if str(row.get("public_access")) == "Yes":
        score += 1.0

    if str(row.get("medical_support_nearby")) == "Yes":
        score += 1.0

    if escalation_label == "Likely escalation":
        if str(row.get("indoor_cooling")) in ["Yes", "Yes - indoor"]:
            score += 2.5
        if str(row.get("water_available")) == "Yes":
            score += 1.5

    elif escalation_label == "Watch":
        if str(row.get("indoor_cooling")) in ["Yes", "Yes - indoor"]:
            score += 1.0

    return score


def recommend_resources(city: str, escalation_label: str, top_n: int = 3) -> pd.DataFrame:
    df = load_resources()
    city_df = df[df["city"] == city].copy()

    if city_df.empty:
        return pd.DataFrame()

    city_df["resource_score"] = city_df.apply(lambda row: score_resource(row, escalation_label), axis=1)

    ranked = city_df.sort_values(
        ["resource_score", "verified_status"],
        ascending=[False, True],
    ).head(top_n)

    return ranked.reset_index(drop=True)