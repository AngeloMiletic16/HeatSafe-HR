from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "best_escalation_72h_model.joblib"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_features.csv"

STABLE_THRESHOLD = 0.30
LIKELY_THRESHOLD = 0.55


@lru_cache(maxsize=1)
def load_escalation_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Escalation model not found: {MODEL_PATH}\n"
            "Run: python -m src.train_escalation_model"
        )
    return joblib.load(MODEL_PATH)


def _get_model_input_columns(model) -> list[str]:
    """
    Iz fitted preprocessor-a izvuče raw input feature stupce
    koje model očekuje.
    """
    preprocessor = model.named_steps["preprocessor"]

    input_columns: list[str] = []
    for _, _, cols in preprocessor.transformers_:
        if cols == "drop":
            continue
        if cols == "passthrough":
            continue
        if isinstance(cols, (list, tuple)):
            input_columns.extend(list(cols))

    return input_columns


def _coerce_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def prepare_escalation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prima DataFrame sa sirovim ili proširenim featureima i vraća
    točno one stupce koje escalation model očekuje, u ispravnom redoslijedu.

    - višak stupaca ignorira
    - ako neki model input stupac nedostaje, dodaje ga kao NaN
    """
    model = load_escalation_model()
    model_input_columns = _get_model_input_columns(model)

    prepared = df.copy()

    for col in model_input_columns:
        if col not in prepared.columns:
            prepared[col] = np.nan

    prepared = prepared[model_input_columns].copy()
    return prepared


def probability_to_label(probability: float) -> str:
    if probability < STABLE_THRESHOLD:
        return "Stable"
    if probability < LIKELY_THRESHOLD:
        return "Watch"
    return "Likely escalation"


def probability_to_flag(probability: float) -> int:
    return int(probability >= LIKELY_THRESHOLD)


def predict_escalation_from_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vraća isti DataFrame + escalation signal stupce:
    - escalation_probability_72h
    - escalation_flag_72h
    - escalation_label_72h
    """
    if feature_df.empty:
        raise ValueError("Input feature DataFrame is empty.")

    model = load_escalation_model()

    meta_df = feature_df.copy()
    model_input_df = prepare_escalation_features(feature_df)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input_df)[:, 1]
    else:
        predictions = model.predict(model_input_df)
        probabilities = predictions.astype(float)

    result_df = meta_df.copy()
    result_df["escalation_probability_72h"] = probabilities
    result_df["escalation_flag_72h"] = result_df["escalation_probability_72h"].apply(probability_to_flag)
    result_df["escalation_label_72h"] = result_df["escalation_probability_72h"].apply(probability_to_label)

    return result_df


def load_feature_dataset() -> pd.DataFrame:
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATA_PATH}\n"
            "Expected existing processed feature file."
        )

    df = pd.read_csv(FEATURE_DATA_PATH)
    df = _coerce_date_column(df)
    return df.sort_values(["city", "date"]).reset_index(drop=True)


def get_latest_city_feature_row(city_name: str) -> pd.DataFrame:
    """
    Dohvaća zadnji raspoloživi feature red za grad iz historical feature dataseta.
    Ovo je dobar prvi korak za testiranje i kasniju integraciju.
    """
    df = load_feature_dataset()

    city_df = df[df["city"] == city_name].copy().sort_values("date")
    if city_df.empty:
        raise ValueError(f"No feature rows found for city: {city_name}")

    latest_row = city_df.tail(1).copy()
    return latest_row

def get_city_feature_row_by_date(city_name: str, target_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Dohvaća feature red za točan grad i datum.
    target_date može biti npr.:
    - "2025-07-15"
    - pd.Timestamp("2025-07-15")
    """
    df = load_feature_dataset()

    target_date = pd.to_datetime(target_date).normalize()

    city_df = df[df["city"] == city_name].copy()
    if city_df.empty:
        raise ValueError(f"No feature rows found for city: {city_name}")

    city_df["date_norm"] = pd.to_datetime(city_df["date"]).dt.normalize()
    match_df = city_df[city_df["date_norm"] == target_date].copy()

    if match_df.empty:
        available_min = city_df["date"].min()
        available_max = city_df["date"].max()
        raise ValueError(
            f"No feature row found for city='{city_name}' and date='{target_date.strftime('%Y-%m-%d')}'. "
            f"Available range: {available_min.strftime('%Y-%m-%d')} to {available_max.strftime('%Y-%m-%d')}."
        )

    return match_df.drop(columns=["date_norm"]).head(1).copy()


def get_city_escalation_snapshot_by_date(city_name: str, target_date: str | pd.Timestamp) -> dict[str, Any]:
    """
    Vraća escalation snapshot za točan grad i datum.
    """
    row_df = get_city_feature_row_by_date(city_name, target_date)
    pred_df = predict_escalation_from_features(row_df)

    row = pred_df.iloc[0]

    return {
        "city": row.get("city"),
        "date": pd.to_datetime(row.get("date")).strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
        "escalation_probability_72h": float(row["escalation_probability_72h"]),
        "escalation_flag_72h": int(row["escalation_flag_72h"]),
        "escalation_label_72h": str(row["escalation_label_72h"]),
    }


def get_city_escalation_summary_by_date(city_name: str, target_date: str | pd.Timestamp) -> dict[str, Any]:
    snapshot = get_city_escalation_snapshot_by_date(city_name, target_date)
    snapshot["operator_message"] = build_escalation_operator_message(
        snapshot["escalation_probability_72h"],
        snapshot["escalation_label_72h"],
    )
    return snapshot


def get_city_escalation_snapshot(city_name: str) -> dict[str, Any]:
    """
    Vraća mali summary dict za grad:
    - city
    - date
    - escalation_probability_72h
    - escalation_flag_72h
    - escalation_label_72h
    """
    latest_row = get_latest_city_feature_row(city_name)
    pred_df = predict_escalation_from_features(latest_row)

    row = pred_df.iloc[0]

    return {
        "city": row.get("city"),
        "date": pd.to_datetime(row.get("date")).strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
        "escalation_probability_72h": float(row["escalation_probability_72h"]),
        "escalation_flag_72h": int(row["escalation_flag_72h"]),
        "escalation_label_72h": str(row["escalation_label_72h"]),
    }


def build_escalation_operator_message(probability: float, label: str) -> str:
    if label == "Stable":
        return (
            f"72h escalation probability is {probability:.2f}. "
            "Short-term escalation signal is low; continue monitoring."
        )
    if label == "Watch":
        return (
            f"72h escalation probability is {probability:.2f}. "
            "Conditions should be watched closely and readiness should be reviewed."
        )
    return (
        f"72h escalation probability is {probability:.2f}. "
        "Likely escalation detected; early preparation and proactive communication are recommended."
    )


def get_city_escalation_summary(city_name: str) -> dict[str, Any]:
    snapshot = get_city_escalation_snapshot(city_name)
    snapshot["operator_message"] = build_escalation_operator_message(
        snapshot["escalation_probability_72h"],
        snapshot["escalation_label_72h"],
    )
    return snapshot


if __name__ == "__main__":
    test_city = "Šibenik"
    test_date = "2025-07-15"

    summary = get_city_escalation_summary_by_date(test_city, test_date)

    print("[TEST] Escalation summary by date")
    for key, value in summary.items():
        print(f"{key}: {value}")