from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src.config import CITIES, MODELS_DIR, PROCESSED_DATA_DIR

RISK_DATA_PATH = PROCESSED_DATA_DIR / "all_cities_daily_with_risk.csv"
STRICT_MODEL_PATH = MODELS_DIR / "best_next_day_risk_model_strict.joblib"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

CLASS_ID_TO_LABEL = {
    0: "Nizak",
    1: "Umjeren",
    2: "Visok",
    3: "Vrlo visok",
}

BASE_WEATHER_COLUMNS = [
    "city",
    "latitude",
    "longitude",
    "date",
    "temp_mean",
    "temp_min",
    "temp_max",
    "humidity_mean",
    "humidity_max",
    "apparent_temp_mean",
    "apparent_temp_max",
    "precipitation_sum",
    "pressure_mean",
    "wind_speed_mean",
    "wind_speed_max",
    "hourly_records",
]


def load_historical_daily_data() -> pd.DataFrame:
    if not RISK_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RISK_DATA_PATH}. Run risk pipeline first."
        )

    df = pd.read_csv(RISK_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)


def load_strict_model():
    if not STRICT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {STRICT_MODEL_PATH}. Run strict model training first."
        )
    return joblib.load(STRICT_MODEL_PATH)


def fetch_daily_forecast(city_name: str, forecast_days: int = 7) -> pd.DataFrame:
    city_info = CITIES[city_name]

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "precipitation_sum",
                "wind_speed_10m_max",
                "relative_humidity_2m_mean",
            ]
        ),
        "timezone": "auto",
        "forecast_days": forecast_days,
    }

    response = requests.get(FORECAST_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "daily" not in payload or "time" not in payload["daily"]:
        raise ValueError(f"No daily forecast returned for {city_name}.")

    df = pd.DataFrame(payload["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df["city"] = city_name

    df = df.rename(
        columns={
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "apparent_temperature_max": "apparent_temp_max",
            "precipitation_sum": "precipitation_sum",
            "wind_speed_10m_max": "wind_speed_max",
            "relative_humidity_2m_mean": "humidity_mean",
        }
    )

    return df[
        [
            "city",
            "date",
            "temp_max",
            "temp_min",
            "apparent_temp_max",
            "precipitation_sum",
            "wind_speed_max",
            "humidity_mean",
        ]
    ].copy()


def build_forecast_base_frame(city_name: str, historical_city_df: pd.DataFrame, forecast_days: int = 7) -> pd.DataFrame:
    forecast_df = fetch_daily_forecast(city_name, forecast_days=forecast_days)

    lat = CITIES[city_name]["lat"]
    lon = CITIES[city_name]["lon"]

    recent_history = historical_city_df.tail(30).copy()

    fallback_pressure = (
        recent_history["pressure_mean"].median()
        if "pressure_mean" in recent_history.columns and not recent_history["pressure_mean"].dropna().empty
        else 1013.25
    )

    fallback_humidity_max_ratio = 1.10
    fallback_wind_ratio = 0.70

    forecast_df["latitude"] = lat
    forecast_df["longitude"] = lon
    forecast_df["temp_mean"] = (forecast_df["temp_max"] + forecast_df["temp_min"]) / 2
    forecast_df["humidity_max"] = (forecast_df["humidity_mean"] * fallback_humidity_max_ratio).clip(upper=100)
    forecast_df["apparent_temp_mean"] = (forecast_df["apparent_temp_max"] + forecast_df["temp_mean"]) / 2
    forecast_df["pressure_mean"] = fallback_pressure
    forecast_df["wind_speed_mean"] = forecast_df["wind_speed_max"] * fallback_wind_ratio
    forecast_df["hourly_records"] = 24

    return forecast_df[BASE_WEATHER_COLUMNS].copy()


def apply_scenario_adjustments(
    forecast_df: pd.DataFrame,
    temperature_delta: float = 0.0,
    humidity_delta: float = 0.0,
    wind_delta: float = 0.0,
) -> pd.DataFrame:
    df = forecast_df.copy()

    df["temp_max"] = df["temp_max"] + temperature_delta
    df["temp_min"] = df["temp_min"] + temperature_delta
    df["temp_mean"] = df["temp_mean"] + temperature_delta
    df["apparent_temp_max"] = df["apparent_temp_max"] + temperature_delta
    df["apparent_temp_mean"] = df["apparent_temp_mean"] + temperature_delta

    df["humidity_mean"] = (df["humidity_mean"] + humidity_delta).clip(lower=0, upper=100)
    df["humidity_max"] = (df["humidity_max"] + humidity_delta).clip(lower=0, upper=100)

    df["wind_speed_max"] = (df["wind_speed_max"] + wind_delta).clip(lower=0)
    df["wind_speed_mean"] = (df["wind_speed_mean"] + wind_delta).clip(lower=0)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)

    return df


def add_base_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["hot_night_flag"] = (df["temp_min"] >= 20).astype(int)
    df["very_hot_day_flag"] = (df["apparent_temp_max"] >= 35).astype(int)
    df["dry_day_flag"] = (df["precipitation_sum"] < 1).astype(int)

    df["hot_day_32"] = (df["apparent_temp_max"] >= 32).astype(int)
    df["very_hot_day_35"] = (df["apparent_temp_max"] >= 35).astype(int)

    return df


def add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["city", "date"]).reset_index(drop=True)

    df["hot_days_last_3"] = (
        df.groupby("city")["hot_day_32"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )

    df["very_hot_days_last_3"] = (
        df.groupby("city")["very_hot_day_35"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lag_columns = [
        "temp_max",
        "temp_min",
        "temp_mean",
        "humidity_mean",
        "apparent_temp_max",
        "precipitation_sum",
        "wind_speed_mean",
    ]

    for col in lag_columns:
        for lag in [1, 2, 3]:
            df[f"{col}_lag_{lag}"] = df.groupby("city")[col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rolling_columns = [
        "temp_max",
        "temp_min",
        "apparent_temp_max",
        "humidity_mean",
    ]

    for col in rolling_columns:
        df[f"{col}_roll3_mean"] = (
            df.groupby("city")[col]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        df[f"{col}_roll7_mean"] = (
            df.groupby("city")[col]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )
        df[f"{col}_roll3_max"] = (
            df.groupby("city")[col]
            .transform(lambda s: s.rolling(3, min_periods=1).max())
        )

    return df


def fill_missing_model_features(future_df: pd.DataFrame, feature_columns: list[str], historical_city_df: pd.DataFrame) -> pd.DataFrame:
    df = future_df.copy()

    numeric_reference = {}
    for col in historical_city_df.columns:
        if pd.api.types.is_numeric_dtype(historical_city_df[col]):
            valid = historical_city_df[col].dropna()
            if not valid.empty:
                numeric_reference[col] = float(valid.median())

    for col in feature_columns:
        if col not in df.columns:
            if col == "city":
                df[col] = df["city"]
            else:
                df[col] = numeric_reference.get(col, 0.0)

    for col in feature_columns:
        if col == "city":
            df[col] = df[col].fillna(df["city"])
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(numeric_reference.get(col, 0.0))

    return df[feature_columns].copy()


def compute_apparent_temp_component(value: float) -> int:
    if value < 26:
        return 0
    if value < 30:
        return 15
    if value < 34:
        return 35
    if value < 38:
        return 55
    return 70


def compute_night_component(value: float) -> int:
    if value < 18:
        return 0
    if value < 22:
        return 5
    if value < 25:
        return 10
    return 15


def compute_humidity_component(value: float) -> int:
    if value < 45:
        return 0
    if value < 60:
        return 4
    if value < 75:
        return 8
    return 10


def compute_wind_component(value: float) -> int:
    if value >= 5:
        return 0
    if value >= 3:
        return 3
    return 6


def compute_precip_adjustment(value: float) -> int:
    if value >= 10:
        return -6
    if value >= 3:
        return -3
    return 0


def assign_risk_label(score: float) -> str:
    if score <= 24:
        return "Nizak"
    if score <= 49:
        return "Umjeren"
    if score <= 74:
        return "Visok"
    return "Vrlo visok"


def add_heuristic_projection(forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()

    df["score_apparent_temp"] = df["apparent_temp_max"].apply(compute_apparent_temp_component)
    df["score_night"] = df["temp_min"].apply(compute_night_component)
    df["score_humidity"] = df["humidity_mean"].apply(compute_humidity_component)
    df["score_wind"] = df["wind_speed_mean"].apply(compute_wind_component)
    df["score_precip_adjustment"] = df["precipitation_sum"].apply(compute_precip_adjustment)
    df["score_persistence"] = (
        (df["hot_days_last_3"] * 3) + (df["very_hot_days_last_3"] * 4)
    ).clip(0, 14)

    df["heuristic_risk_score"] = (
        df["score_apparent_temp"]
        + df["score_night"]
        + df["score_humidity"]
        + df["score_wind"]
        + df["score_precip_adjustment"]
        + df["score_persistence"]
    ).clip(0, 100)

    df["heuristic_risk_level"] = df["heuristic_risk_score"].apply(assign_risk_label)
    return df


def make_ml_forecast(
    city_name: str,
    forecast_days: int = 7,
    temperature_delta: float = 0.0,
    humidity_delta: float = 0.0,
    wind_delta: float = 0.0,
) -> pd.DataFrame:
    historical_df = load_historical_daily_data()
    historical_city_df = historical_df[historical_df["city"] == city_name].sort_values("date").copy()

    if historical_city_df.empty:
        raise ValueError(f"No historical data found for city {city_name}")

    forecast_df = build_forecast_base_frame(
        city_name=city_name,
        historical_city_df=historical_city_df,
        forecast_days=forecast_days,
    )
    forecast_df = apply_scenario_adjustments(
        forecast_df,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
    forecast_df["is_forecast"] = 1

    history_seed = historical_city_df[BASE_WEATHER_COLUMNS].tail(30).copy()
    history_seed["is_forecast"] = 0

    combined = pd.concat([history_seed, forecast_df], ignore_index=True)
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)

    combined = add_calendar_features(combined)
    combined = add_base_derived_features(combined)
    combined = add_persistence_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)

    future_df = combined[combined["is_forecast"] == 1].copy().reset_index(drop=True)
    future_df = add_heuristic_projection(future_df)

    model = load_strict_model()
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Strict model does not expose feature_names_in_")

    feature_columns = list(model.feature_names_in_)
    x_future = fill_missing_model_features(future_df, feature_columns, historical_city_df)

    predictions = model.predict(x_future)
    probabilities = model.predict_proba(x_future)
    confidence = probabilities.max(axis=1)

    future_df["ml_predicted_class"] = predictions
    future_df["ml_predicted_label"] = future_df["ml_predicted_class"].map(CLASS_ID_TO_LABEL)
    future_df["ml_prediction_confidence"] = confidence

    for idx, class_id in enumerate(model.classes_):
        future_df[f"proba_class_{class_id}"] = probabilities[:, idx]

    return future_df.sort_values("date").reset_index(drop=True)