from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCES_PATH = PROJECT_ROOT / "data" / "resources" / "cooling_centers.csv"
CRITICAL_POINTS_PATH = PROJECT_ROOT / "data" / "resources" / "critical_points.csv"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _normalize_yes_no(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    positive = {"yes", "true", "1", "da", "available", "verified", "y"}
    negative = {"no", "false", "0", "ne", "unavailable", "n"}

    if text in positive:
        return "Yes"
    if text in negative:
        return "No"

    if "yes" in text or "da" in text:
        return "Yes"
    if "no" in text or "ne" in text:
        return "No"

    return str(value)


def _normalize_verified(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()
    if "verified" in text:
        return "Verified"
    if "demo" in text:
        return "Demo"
    if "partial" in text:
        return "Partial"
    return str(value)


def _ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "lat" not in out.columns and "latitude" in out.columns:
        out["lat"] = out["latitude"]
    if "lon" not in out.columns and "longitude" in out.columns:
        out["lon"] = out["longitude"]

    return out


def load_resources() -> pd.DataFrame:
    if not RESOURCES_PATH.exists():
        raise FileNotFoundError(f"Missing resources file: {RESOURCES_PATH}")

    df = pd.read_csv(RESOURCES_PATH)
    df = _ensure_lat_lon(df)

    required = ["city", "resource_name", "resource_type", "address"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required resource columns: {missing_required}")

    optional_defaults = {
        "hours_weekday": "Unknown",
        "verified_status": "Demo",
        "water_available": "Unknown",
        "indoor_cooling": "Unknown",
        "lat": None,
        "lon": None,
        "capacity_estimate": 100,
        "current_occupancy_pct": 35,
        "readiness_score": 70,
        "dispatch_priority": 50,
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["verified_status"] = df["verified_status"].apply(_normalize_verified)
    df["water_available"] = df["water_available"].apply(_normalize_yes_no)
    df["indoor_cooling"] = df["indoor_cooling"].apply(_normalize_yes_no)

    df["capacity_estimate"] = df["capacity_estimate"].apply(lambda x: _safe_float(x, 100))
    df["current_occupancy_pct"] = df["current_occupancy_pct"].apply(lambda x: _safe_float(x, 35))
    df["readiness_score"] = df["readiness_score"].apply(lambda x: _safe_float(x, 70))
    df["dispatch_priority"] = df["dispatch_priority"].apply(lambda x: _safe_float(x, 50))

    return df.copy()


def load_critical_points() -> pd.DataFrame:
    if not CRITICAL_POINTS_PATH.exists():
        return pd.DataFrame(columns=["city", "point_name", "point_type", "lat", "lon"])

    df = pd.read_csv(CRITICAL_POINTS_PATH)
    df = _ensure_lat_lon(df)

    required = ["city", "point_name", "point_type", "lat", "lon"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required critical point columns: {missing_required}")

    return df.copy()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def compute_opening_score(hours_text: Any) -> float:
    if pd.isna(hours_text):
        return 60.0

    text = str(hours_text).strip().lower()

    if "24/7" in text or "24h" in text:
        return 100.0
    if "unknown" in text or text == "":
        return 60.0
    if "closed" in text or "zatvor" in text:
        return 0.0

    return 75.0


def compute_capacity_score(row: pd.Series) -> float:
    occupancy_pct = _safe_float(row.get("current_occupancy_pct"), 35.0)
    return _clamp_0_100(100.0 - occupancy_pct)


def compute_trust_score(row: pd.Series) -> float:
    verified = str(row.get("verified_status", "Unknown"))

    if verified == "Verified":
        return 100.0
    if verified == "Partial":
        return 75.0
    if verified == "Demo":
        return 50.0
    return 60.0


def compute_amenity_score(row: pd.Series) -> float:
    score = 50.0

    if str(row.get("indoor_cooling", "Unknown")) == "Yes":
        score += 30.0
    if str(row.get("water_available", "Unknown")) == "Yes":
        score += 20.0

    return _clamp_0_100(score)


def compute_proximity_to_critical_points(
    resource_row: pd.Series,
    critical_points_df: pd.DataFrame,
) -> tuple[float, str, float]:
    city_points = critical_points_df[critical_points_df["city"] == resource_row["city"]].copy()

    lat = resource_row.get("lat")
    lon = resource_row.get("lon")

    if city_points.empty or pd.isna(lat) or pd.isna(lon):
        return 50.0, "N/A", None

    nearest_name = "N/A"
    nearest_distance = None

    for _, point in city_points.iterrows():
        if pd.isna(point["lat"]) or pd.isna(point["lon"]):
            continue

        dist = haversine_km(
            float(lat),
            float(lon),
            float(point["lat"]),
            float(point["lon"]),
        )

        if nearest_distance is None or dist < nearest_distance:
            nearest_distance = dist
            nearest_name = str(point["point_name"])

    if nearest_distance is None:
        return 50.0, "N/A", None

    # 0 km -> 100, 2 km -> ~67, 5 km -> ~37, 8 km -> ~20
    proximity_score = 100.0 * math.exp(-nearest_distance / 5.0)
    return _clamp_0_100(proximity_score), nearest_name, round(nearest_distance, 2)


def compute_priority_fit_score(
    resource_row: pd.Series,
    escalation_label: str,
    priority_groups: list[str] | None = None,
) -> float:
    resource_type = str(resource_row.get("resource_type", "")).lower()
    priority_groups = priority_groups or []

    score = 55.0

    if "hospital" in resource_type or "health" in resource_type:
        score += 18.0

    if "public indoor" in resource_type or "library" in resource_type or "sports hall" in resource_type:
        score += 14.0

    if "water" in resource_type:
        score += 8.0

    if "tourist" in resource_type and any("tourist" in g.lower() for g in priority_groups):
        score += 12.0

    if any("older" in g.lower() or "elderly" in g.lower() for g in priority_groups):
        if "hospital" in resource_type or "health" in resource_type or "public indoor" in resource_type:
            score += 10.0

    if escalation_label == "Likely escalation":
        score += 6.0

    return _clamp_0_100(score)


def build_dispatch_reason(row: pd.Series) -> str:
    reasons = []

    if _safe_float(row.get("readiness_score"), 0) >= 80:
        reasons.append("high operational readiness")
    if _safe_float(row.get("capacity_availability_score"), 0) >= 70:
        reasons.append("good available capacity")
    if _safe_float(row.get("proximity_score"), 0) >= 70:
        reasons.append("close to critical points")
    if str(row.get("verified_status", "")) == "Verified":
        reasons.append("verified location")
    if str(row.get("indoor_cooling", "")) == "Yes":
        reasons.append("indoor cooling available")
    if str(row.get("water_available", "")) == "Yes":
        reasons.append("water available")

    if not reasons:
        reasons.append("balanced fallback operational option")

    return ", ".join(reasons[:4])


def recommend_dispatch_resources(
    city: str,
    escalation_label: str = "Stable",
    priority_groups: list[str] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    resources_df = load_resources()
    critical_points_df = load_critical_points()

    city_df = resources_df[resources_df["city"] == city].copy()
    if city_df.empty:
        return pd.DataFrame()

    scored_rows = []

    for _, row in city_df.iterrows():
        opening_score = compute_opening_score(row.get("hours_weekday"))
        capacity_score = compute_capacity_score(row)
        trust_score = compute_trust_score(row)
        amenity_score = compute_amenity_score(row)
        readiness_score = _clamp_0_100(_safe_float(row.get("readiness_score"), 70.0))
        dispatch_priority = _clamp_0_100(_safe_float(row.get("dispatch_priority"), 50.0))

        proximity_score, nearest_point, nearest_distance_km = compute_proximity_to_critical_points(
            row,
            critical_points_df,
        )

        priority_fit_score = compute_priority_fit_score(
            row,
            escalation_label=escalation_label,
            priority_groups=priority_groups,
        )

        dispatch_score = (
            0.24 * readiness_score
            + 0.18 * capacity_score
            + 0.18 * proximity_score
            + 0.12 * opening_score
            + 0.10 * trust_score
            + 0.10 * amenity_score
            + 0.08 * priority_fit_score
            + 0.10 * dispatch_priority
        )

        scored_rows.append(
            {
                **row.to_dict(),
                "opening_score": round(opening_score, 2),
                "capacity_availability_score": round(capacity_score, 2),
                "trust_score": round(trust_score, 2),
                "amenity_score": round(amenity_score, 2),
                "priority_fit_score": round(priority_fit_score, 2),
                "proximity_score": round(proximity_score, 2),
                "nearest_critical_point": nearest_point,
                "nearest_critical_distance_km": nearest_distance_km,
                "dispatch_score": round(dispatch_score, 2),
            }
        )

    result_df = pd.DataFrame(scored_rows).sort_values(
        ["dispatch_score", "readiness_score", "capacity_availability_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    result_df["dispatch_rank"] = range(1, len(result_df) + 1)
    result_df["dispatch_reason"] = result_df.apply(build_dispatch_reason, axis=1)

    return result_df.head(top_n).copy()


def build_top_dispatch_summary(dispatch_df: pd.DataFrame) -> str:
    if dispatch_df.empty:
        return "No dispatch resource recommendation available."

    row = dispatch_df.iloc[0]

    distance_text = "N/A"
    if pd.notna(row.get("nearest_critical_distance_km")):
        distance_text = f"{float(row['nearest_critical_distance_km']):.2f} km"

    return (
        f"Top dispatch resource: {row.get('resource_name', 'Unknown')} | "
        f"Type: {row.get('resource_type', 'N/A')} | "
        f"Dispatch score: {float(row.get('dispatch_score', 0)):.1f} | "
        f"Nearest critical point: {row.get('nearest_critical_point', 'N/A')} | "
        f"Distance: {distance_text} | "
        f"Reason: {row.get('dispatch_reason', 'N/A')}"
    )


if __name__ == "__main__":
    test = recommend_dispatch_resources(
        city="Šibenik",
        escalation_label="Likely escalation",
        priority_groups=[
            "Older adults and chronically ill residents",
            "Tourists and event visitors",
        ],
        top_n=5,
    )
    print(test[["dispatch_rank", "resource_name", "dispatch_score", "nearest_critical_point", "nearest_critical_distance_km", "dispatch_reason"]])
    print()
    print(build_top_dispatch_summary(test))