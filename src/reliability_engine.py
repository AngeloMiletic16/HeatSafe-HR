from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary
from src.escalation_engine import predict_escalation_from_features
from src.forecast_engine import make_ml_forecast
from src.vulnerability_engine import (
    build_impact_adjusted_priority,
    get_city_vulnerability_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STRICT_METRICS_PATH = MODELS_DIR / "model_metrics_strict.json"

STRICT_MODEL_CANDIDATES = [
    MODELS_DIR / "best_model_strict.joblib",
    MODELS_DIR / "best_strict_model.joblib",
    MODELS_DIR / "strict_model.joblib",
    MODELS_DIR / "best_model_v2.joblib",
    MODELS_DIR / "random_forest_strict.joblib",
]

REFERENCE_DATASET_CANDIDATES = [
    PROCESSED_DIR / "all_cities_daily_with_risk.csv",
    PROCESSED_DIR / "model_dataset.csv",
    PROCESSED_DIR / "model_dataset_strict.csv",
]

RISK_ORDER = {
    "Nizak": 0,
    "Umjeren": 1,
    "Visok": 2,
    "Vrlo visok": 3,
}

V3_ORDER = {
    "Stable": 0,
    "Watch": 1,
    "Likely escalation": 2,
}

DROP_REFERENCE_COLUMNS = {
    "date",
    "risk_level",
    "target",
    "target_label",
    "target_escalation_72h",
    "predicted_label",
    "ml_predicted_label",
    "correct_prediction",
    "true_label",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_strict_model_path() -> Path | None:
    metrics_json = _load_json_if_exists(STRICT_METRICS_PATH)
    best_model = metrics_json.get("best_model")

    dynamic_candidates: list[Path] = []
    if best_model:
        dynamic_candidates.extend(
            [
                MODELS_DIR / f"{best_model}.joblib",
                MODELS_DIR / f"best_{best_model}.joblib",
            ]
        )

    for candidate in dynamic_candidates + STRICT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def _load_reference_dataset() -> pd.DataFrame:
    for candidate in REFERENCE_DATASET_CANDIDATES:
        if candidate.exists():
            return pd.read_csv(candidate)

    raise FileNotFoundError(
        "No reference dataset found for reliability engine. "
        f"Checked: {[str(p) for p in REFERENCE_DATASET_CANDIDATES]}"
    )


def _reference_feature_columns(reference_df: pd.DataFrame) -> list[str]:
    return [c for c in reference_df.columns if c not in DROP_REFERENCE_COLUMNS]


def _fill_value(series: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        return series.median()
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return None


def _prepare_input_row(row_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = _reference_feature_columns(reference_df)
    prepared = pd.DataFrame(index=[0])

    for col in feature_cols:
        if col in row_df.columns:
            prepared[col] = [row_df.iloc[0][col]]
        else:
            prepared[col] = [_fill_value(reference_df[col])]

    return prepared


def _unwrap_model_bundle(model_obj: Any) -> tuple[Any | None, Any]:
    if isinstance(model_obj, Pipeline):
        return model_obj, model_obj.steps[-1][1]

    if isinstance(model_obj, dict):
        for key in ["pipeline", "model_pipeline", "best_pipeline", "clf_pipeline"]:
            candidate = model_obj.get(key)
            if isinstance(candidate, Pipeline):
                return candidate, candidate.steps[-1][1]

        for key in ["model", "estimator", "clf", "best_model"]:
            candidate = model_obj.get(key)
            if candidate is not None:
                return None, candidate

    return None, model_obj


def _predict_with_strict_model(prepared_row: pd.DataFrame) -> dict[str, Any]:
    model_path = _find_strict_model_path()
    if model_path is None:
        return {
            "label": None,
            "confidence": None,
            "available": False,
            "warning": "Strict model file not found.",
        }

    try:
        model_obj = joblib.load(model_path)
        pipeline, estimator = _unwrap_model_bundle(model_obj)

        if pipeline is not None:
            pred = pipeline.predict(prepared_row)[0]
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(prepared_row)[0]
                confidence = float(max(proba))
            else:
                confidence = None
        else:
            if hasattr(estimator, "feature_names_in_"):
                model_cols = list(estimator.feature_names_in_)
                x = prepared_row[model_cols].copy()
            else:
                numeric_cols = prepared_row.select_dtypes(include=["number"]).columns.tolist()
                x = prepared_row[numeric_cols].copy()

            pred = estimator.predict(x)[0]
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(x)[0]
                confidence = float(max(proba))
            else:
                confidence = None

        return {
            "label": str(pred),
            "confidence": confidence,
            "available": True,
            "warning": "",
        }

    except Exception as exc:
        return {
            "label": None,
            "confidence": None,
            "available": False,
            "warning": f"Strict model prediction failed: {exc}",
        }


def _normalize_v1_signal(first_row: pd.Series) -> tuple[str | None, float | None]:
    label = first_row.get("ml_predicted_label")
    confidence = first_row.get("ml_prediction_confidence")

    if pd.isna(label):
        label = first_row.get("heuristic_risk_level")

    if pd.isna(label):
        return None, None

    return str(label), _safe_float(confidence, default=None) if confidence is not None else None


def _normalize_v3_signal(city: str, first_row_df: pd.DataFrame) -> tuple[str | None, float | None]:
    if "city" not in first_row_df.columns:
        first_row_df = first_row_df.copy()
        first_row_df["city"] = city

    pred_df = predict_escalation_from_features(first_row_df)
    row = pred_df.iloc[0]

    return (
        str(row["escalation_label_72h"]),
        float(row["escalation_probability_72h"]),
    )


def _v3_certainty(probability: float | None) -> float:
    if probability is None:
        return 0.35

    p = float(probability)

    if p <= 0.20:
        return 0.95
    if p <= 0.30:
        return 0.80
    if p < 0.55:
        return 0.40
    if p < 0.70:
        return 0.70
    return 0.90


def _expected_v3_from_risk(v1_signal: str | None, v2_signal: str | None, peak_signal: str | None) -> int:
    ranks = []

    if v1_signal in RISK_ORDER:
        ranks.append(RISK_ORDER[v1_signal])
    if v2_signal in RISK_ORDER:
        ranks.append(RISK_ORDER[v2_signal])
    if peak_signal in RISK_ORDER:
        ranks.append(RISK_ORDER[peak_signal])

    if not ranks:
        return 1

    max_rank = max(ranks)

    if max_rank <= 0:
        return 0
    if max_rank == 1:
        return 1
    return 2


def _compute_consensus_status(
    v1_signal: str | None,
    v2_signal: str | None,
    v3_signal: str | None,
    peak_signal: str | None,
) -> tuple[str, float]:
    if v1_signal not in RISK_ORDER and v2_signal not in RISK_ORDER:
        return "Insufficient comparison", 25.0

    if v1_signal in RISK_ORDER and v2_signal in RISK_ORDER:
        v1_v2_diff = abs(RISK_ORDER[v1_signal] - RISK_ORDER[v2_signal])
    else:
        v1_v2_diff = 1

    expected_v3_rank = _expected_v3_from_risk(v1_signal, v2_signal, peak_signal)
    actual_v3_rank = V3_ORDER.get(str(v3_signal), 1)
    v3_diff = abs(expected_v3_rank - actual_v3_rank)

    if v1_signal in RISK_ORDER and v2_signal in RISK_ORDER and v1_v2_diff == 0 and v3_diff == 0:
        return "Strong consensus", 95.0

    if v1_v2_diff <= 1 and v3_diff <= 1:
        return "Moderate consensus", 72.0

    if v1_v2_diff <= 1 and v3_diff >= 2:
        return "Mixed signals", 48.0

    return "Low consensus", 25.0


def _confidence_level(
    consensus_score: float,
    v1_confidence: float | None,
    v2_confidence: float | None,
    v3_probability: float | None,
) -> tuple[str, float]:
    v1_score = 100.0 * _clamp_0_1(v1_confidence) if v1_confidence is not None else 55.0
    v2_score = 100.0 * _clamp_0_1(v2_confidence) if v2_confidence is not None else 45.0
    v3_score = 100.0 * _v3_certainty(v3_probability)

    reliability_score = round(
        0.45 * consensus_score + 0.20 * v1_score + 0.15 * v2_score + 0.20 * v3_score,
        2,
    )

    if reliability_score >= 80:
        return "High", reliability_score
    if reliability_score >= 60:
        return "Moderate", reliability_score
    return "Low", reliability_score


def _build_uncertainty_warning(
    strict_available: bool,
    strict_warning: str,
    consensus_status: str,
    confidence_level: str,
    v3_signal: str | None,
    impact_adjusted_priority: float,
) -> tuple[str, bool]:
    if not strict_available:
        warning = f"Strict model signal unavailable. {strict_warning}".strip()
        review_required = impact_adjusted_priority >= 45
        return warning, review_required

    if consensus_status == "Low consensus":
        return (
            "Low consensus between models. Operator review is recommended before escalating action.",
            True,
        )

    if consensus_status == "Mixed signals":
        return (
            "Multiclass heat-risk signals and escalation early-warning signal are not fully aligned.",
            True,
        )

    if confidence_level == "Low":
        return (
            "Overall reliability is low due to disagreement and/or borderline confidence signals.",
            True,
        )

    if str(v3_signal) == "Watch" and impact_adjusted_priority >= 50:
        return (
            "Watch-zone escalation combined with elevated city impact priority suggests closer operator monitoring.",
            True,
        )

    return "No major uncertainty warning.", False


def build_reliability_snapshot(
    city: str,
    temperature_delta: float = 0.0,
    humidity_delta: float = 0.0,
    wind_delta: float = 0.0,
) -> dict[str, Any]:
    forecast_df = make_ml_forecast(
        city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )

    if "city" not in forecast_df.columns:
        forecast_df["city"] = city

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    first_row_df = forecast_df.sort_values("date").head(1).copy()
    first_row = first_row_df.iloc[0]

    summary = build_city_readiness_summary(city, forecast_df)
    vulnerability_snapshot = get_city_vulnerability_snapshot(city)

    v1_signal, v1_confidence = _normalize_v1_signal(first_row)
    v3_signal, v3_probability = _normalize_v3_signal(city, first_row_df)

    reference_df = _load_reference_dataset()
    prepared_row = _prepare_input_row(first_row_df, reference_df)
    v2_result = _predict_with_strict_model(prepared_row)

    v2_signal = v2_result["label"]
    v2_confidence = v2_result["confidence"]
    strict_available = bool(v2_result["available"])

    impact_adjusted_priority = build_impact_adjusted_priority(
        next_7d_peak_score=float(summary["next_7d_peak_score"]),
        escalation_probability_72h=v3_probability,
        vulnerability_index=float(vulnerability_snapshot["vulnerability_index"]),
    )

    consensus_status, consensus_score = _compute_consensus_status(
        v1_signal=v1_signal,
        v2_signal=v2_signal,
        v3_signal=v3_signal,
        peak_signal=summary["next_7d_peak_level"],
    )

    confidence_level, reliability_score = _confidence_level(
        consensus_score=consensus_score,
        v1_confidence=v1_confidence,
        v2_confidence=v2_confidence,
        v3_probability=v3_probability,
    )

    uncertainty_warning, operator_review_required = _build_uncertainty_warning(
        strict_available=strict_available,
        strict_warning=str(v2_result.get("warning", "")),
        consensus_status=consensus_status,
        confidence_level=confidence_level,
        v3_signal=v3_signal,
        impact_adjusted_priority=impact_adjusted_priority,
    )

    return {
        "city": city,
        "date": pd.to_datetime(first_row["date"]),
        "v1_signal": v1_signal,
        "v1_confidence": v1_confidence,
        "v2_signal": v2_signal,
        "v2_confidence": v2_confidence,
        "v3_signal": v3_signal,
        "v3_probability": v3_probability,
        "next_24h_risk": summary["next_24h_level"],
        "next_7d_peak_level": summary["next_7d_peak_level"],
        "next_7d_peak_score": float(summary["next_7d_peak_score"]),
        "readiness_status": summary["readiness_status"],
        "vulnerability_index": float(vulnerability_snapshot["vulnerability_index"]),
        "vulnerability_band": vulnerability_snapshot["vulnerability_band"],
        "impact_adjusted_priority": float(impact_adjusted_priority),
        "consensus_status": consensus_status,
        "confidence_level": confidence_level,
        "consensus_score": float(consensus_score),
        "reliability_score": float(reliability_score),
        "uncertainty_warning": uncertainty_warning,
        "operator_review_required": bool(operator_review_required),
        "strict_model_available": strict_available,
        "strict_model_warning": str(v2_result.get("warning", "")),
    }


def build_multi_city_reliability_table(
    cities: list[str],
    temperature_delta: float = 0.0,
    humidity_delta: float = 0.0,
    wind_delta: float = 0.0,
) -> pd.DataFrame:
    rows = []

    for city in cities:
        rows.append(
            build_reliability_snapshot(
                city=city,
                temperature_delta=temperature_delta,
                humidity_delta=humidity_delta,
                wind_delta=wind_delta,
            )
        )

    return pd.DataFrame(rows)


def build_system_health_summary(reliability_df: pd.DataFrame) -> dict[str, Any]:
    if reliability_df.empty:
        return {
            "system_health": "No data",
            "cities_total": 0,
            "strong_consensus_count": 0,
            "low_consensus_count": 0,
            "operator_review_count": 0,
            "avg_reliability_score": None,
        }

    strong_consensus_count = int((reliability_df["consensus_status"] == "Strong consensus").sum())
    low_consensus_count = int(
        reliability_df["consensus_status"].isin(["Low consensus", "Mixed signals"]).sum()
    )
    operator_review_count = int(reliability_df["operator_review_required"].sum())
    avg_reliability_score = round(float(reliability_df["reliability_score"].mean()), 2)

    if operator_review_count == 0 and avg_reliability_score >= 80:
        system_health = "Healthy"
    elif operator_review_count <= max(1, len(reliability_df) // 3) and avg_reliability_score >= 65:
        system_health = "Watch"
    else:
        system_health = "Degraded"

    return {
        "system_health": system_health,
        "cities_total": int(len(reliability_df)),
        "strong_consensus_count": strong_consensus_count,
        "low_consensus_count": low_consensus_count,
        "operator_review_count": operator_review_count,
        "avg_reliability_score": avg_reliability_score,
    }


if __name__ == "__main__":
    demo_cities = [
        "Dubrovnik",
        "Osijek",
        "Rijeka",
        "Split",
        "Šibenik",
        "Zadar",
        "Zagreb",
    ]

    table = build_multi_city_reliability_table(
        demo_cities,
        temperature_delta=6,
        humidity_delta=10,
        wind_delta=-3,
    )

    health = build_system_health_summary(table)

    print("[TEST] Reliability table")
    print(
        table[
            [
                "city",
                "v1_signal",
                "v2_signal",
                "v3_signal",
                "consensus_status",
                "confidence_level",
                "impact_adjusted_priority",
                "operator_review_required",
            ]
        ]
    )
    print()
    print("[TEST] System health")
    print(health)