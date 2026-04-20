from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.escalation_engine import predict_escalation_from_features

try:
    import shap  # type: ignore
except Exception:
    shap = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESCALATION_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "best_escalation_72h_model.joblib"
ESCALATION_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset_escalation_72h.csv"

TARGET_COL = "target_escalation_72h"
DROP_FOR_REFERENCE = {TARGET_COL}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_model_object() -> Any:
    if not ESCALATION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing escalation model: {ESCALATION_MODEL_PATH}")
    return joblib.load(ESCALATION_MODEL_PATH)


def _load_reference_dataset() -> pd.DataFrame:
    if not ESCALATION_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing escalation dataset: {ESCALATION_DATASET_PATH}")

    df = pd.read_csv(ESCALATION_DATASET_PATH)
    return df.copy()


def _unwrap_model_bundle(model_obj: Any) -> tuple[Any | None, Any]:
    """
    Vraća:
    - pipeline (ako postoji)
    - estimator
    """
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


def _feature_columns(reference_df: pd.DataFrame) -> list[str]:
    return [c for c in reference_df.columns if c not in DROP_FOR_REFERENCE]


def _fill_value(series: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        return series.median()
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return None


def _prepare_input_row(row_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    row_df = row_df.copy()
    feature_cols = _feature_columns(reference_df)

    prepared = pd.DataFrame(index=[0])

    for col in feature_cols:
        if col in row_df.columns:
            prepared[col] = [row_df.iloc[0][col]]
        else:
            prepared[col] = [_fill_value(reference_df[col])]

    return prepared


def _sample_background(reference_df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    feature_cols = _feature_columns(reference_df)
    sample_n = min(n, len(reference_df))
    return reference_df[feature_cols].sample(sample_n, random_state=42).reset_index(drop=True)


def _maybe_dense(x: Any) -> Any:
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def _clean_feature_name(name: str) -> str:
    cleaned = str(name)
    prefixes = [
        "num__",
        "cat__",
        "remainder__",
        "onehot__",
        "ordinal__",
        "passthrough__",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    cleaned = cleaned.replace("_", " ")
    return cleaned


def _extract_preprocessed_data(
    pipeline: Pipeline,
    input_row: pd.DataFrame,
    background_df: pd.DataFrame,
) -> tuple[Any, Any, list[str], Any]:
    preprocessor = pipeline[:-1]
    estimator = pipeline.steps[-1][1]

    x_trans = preprocessor.transform(input_row)
    bg_trans = preprocessor.transform(background_df)

    x_trans = _maybe_dense(x_trans)
    bg_trans = _maybe_dense(bg_trans)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        if hasattr(estimator, "feature_names_in_"):
            feature_names = list(estimator.feature_names_in_)
        else:
            width = x_trans.shape[1]
            feature_names = [f"feature_{i}" for i in range(width)]

    return x_trans, bg_trans, feature_names, estimator


def _extract_direct_estimator_data(
    estimator: Any,
    input_row: pd.DataFrame,
    background_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], Any]:
    if hasattr(estimator, "feature_names_in_"):
        cols = [c for c in estimator.feature_names_in_ if c in input_row.columns]
        if not cols:
            raise ValueError("Estimator has feature_names_in_, but no matching columns were found in input row.")

        x_df = input_row[cols].copy()
        bg_df = background_df[cols].copy()
        feature_names = cols
    else:
        numeric_cols = input_row.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available for direct estimator explanation.")
        x_df = input_row[numeric_cols].copy()
        bg_df = background_df[numeric_cols].copy()
        feature_names = numeric_cols

    x_arr = x_df.to_numpy()
    bg_arr = bg_df.to_numpy()

    return x_arr, bg_arr, feature_names, estimator


def _select_positive_class_shap(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1])[0]
        return np.asarray(shap_values[0])[0]

    arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # shape: (samples, features, classes)
        if arr.shape[-1] >= 2:
            return arr[0, :, 1]
        return arr[0, :, 0]

    if arr.ndim == 2:
        return arr[0]

    raise ValueError("Unsupported SHAP values shape.")


def _fallback_local_explanation(row_df: pd.DataFrame, probability: float, label: str) -> dict[str, Any]:
    row = row_df.iloc[0]

    drivers: list[tuple[str, float]] = []
    protective: list[tuple[str, float]] = []

    apparent_temp = _safe_float(row.get("apparent_temp_max"), np.nan)
    temp_max = _safe_float(row.get("temp_max"), np.nan)
    humidity = _safe_float(row.get("humidity_mean"), np.nan)
    wind = _safe_float(row.get("wind_speed_mean"), np.nan)
    temp_min = _safe_float(row.get("temp_min"), np.nan)

    if not np.isnan(apparent_temp):
        if apparent_temp >= 28:
            drivers.append(("Higher apparent temperature", apparent_temp))
        elif apparent_temp <= 22:
            protective.append(("Lower apparent temperature", apparent_temp))

    if not np.isnan(temp_max):
        if temp_max >= 30:
            drivers.append(("Higher maximum temperature", temp_max))
        elif temp_max <= 24:
            protective.append(("Lower maximum temperature", temp_max))

    if not np.isnan(wind):
        if wind <= 3:
            drivers.append(("Lower wind speed", wind))
        elif wind >= 8:
            protective.append(("Stronger wind ventilation", wind))

    if not np.isnan(humidity):
        if humidity >= 70:
            drivers.append(("Higher humidity load", humidity))
        elif humidity <= 45:
            protective.append(("Lower humidity load", humidity))

    if not np.isnan(temp_min):
        if temp_min >= 20:
            drivers.append(("Warmer night-time conditions", temp_min))
        elif temp_min <= 15:
            protective.append(("Cooler night-time recovery", temp_min))

    drivers = sorted(drivers, key=lambda x: abs(x[1]), reverse=True)[:4]
    protective = sorted(protective, key=lambda x: abs(x[1]), reverse=True)[:4]

    if drivers:
        driver_text = ", ".join(name for name, _ in drivers[:3])
    else:
        driver_text = "no strong local heat-pressure drivers detected"

    explanation_text = (
        f"V3 model gives 72h escalation probability {probability:.2f} and label '{label}'. "
        f"Fallback explanation suggests the main local drivers are: {driver_text}."
    )

    return {
        "method": "heuristic_fallback",
        "probability": probability,
        "label": label,
        "top_positive_drivers": [{"feature": name, "contribution": value} for name, value in drivers],
        "top_protective_drivers": [{"feature": name, "contribution": value} for name, value in protective],
        "explanation_text": explanation_text,
    }


def explain_escalation_row(row_df: pd.DataFrame, top_n: int = 4) -> dict[str, Any]:
    """
    Vraća lokalno objašnjenje za jedan red (tipično prvi forecast dan).
    Koristi SHAP ako je dostupan i ako model bundle to dopušta.
    U suprotnom pada na heuristički fallback.
    """
    if row_df.empty:
        return {
            "method": "none",
            "probability": None,
            "label": None,
            "top_positive_drivers": [],
            "top_protective_drivers": [],
            "explanation_text": "No row available for explanation.",
        }

    prediction_df = predict_escalation_from_features(row_df.copy())
    pred_row = prediction_df.iloc[0]
    probability = float(pred_row["escalation_probability_72h"])
    label = str(pred_row["escalation_label_72h"])

    if shap is None:
        return _fallback_local_explanation(row_df, probability, label)

    try:
        model_obj = _load_model_object()
        reference_df = _load_reference_dataset()

        pipeline, estimator = _unwrap_model_bundle(model_obj)
        input_row = _prepare_input_row(row_df, reference_df)
        background_df = _sample_background(reference_df, n=200)

        if pipeline is not None:
            x_trans, bg_trans, feature_names, estimator = _extract_preprocessed_data(
                pipeline,
                input_row,
                background_df,
            )
        else:
            x_trans, bg_trans, feature_names, estimator = _extract_direct_estimator_data(
                estimator,
                input_row,
                background_df,
            )

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(x_trans)
        selected_values = _select_positive_class_shap(shap_values)

        contrib_df = pd.DataFrame(
            {
                "feature": [_clean_feature_name(f) for f in feature_names],
                "shap_value": selected_values,
            }
        )
        contrib_df["abs_shap"] = contrib_df["shap_value"].abs()

        positive_df = (
            contrib_df[contrib_df["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(top_n)
        )
        protective_df = (
            contrib_df[contrib_df["shap_value"] < 0]
            .sort_values("shap_value", ascending=True)
            .head(top_n)
        )

        top_positive = [
            {
                "feature": row["feature"],
                "contribution": round(float(row["shap_value"]), 4),
            }
            for _, row in positive_df.iterrows()
        ]

        top_protective = [
            {
                "feature": row["feature"],
                "contribution": round(float(row["shap_value"]), 4),
            }
            for _, row in protective_df.iterrows()
        ]

        if top_positive:
            driver_text = ", ".join(item["feature"] for item in top_positive[:3])
        else:
            driver_text = "no dominant positive drivers detected"

        explanation_text = (
            f"V3 model gives 72h escalation probability {probability:.2f} and label '{label}'. "
            f"SHAP local explanation shows the strongest positive drivers are: {driver_text}."
        )

        return {
            "method": "shap_tree",
            "probability": probability,
            "label": label,
            "top_positive_drivers": top_positive,
            "top_protective_drivers": top_protective,
            "explanation_text": explanation_text,
        }

    except Exception:
        return _fallback_local_explanation(row_df, probability, label)


if __name__ == "__main__":
    test_row = pd.DataFrame(
        [
            {
                "city": "Šibenik",
                "temp_max": 31.0,
                "apparent_temp_max": 33.0,
                "humidity_mean": 58.0,
                "wind_speed_mean": 2.5,
                "temp_min": 21.0,
            }
        ]
    )
    print(explain_escalation_row(test_row))