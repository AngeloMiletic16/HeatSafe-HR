from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset_escalation_72h.csv"

MODELS_DIR = PROJECT_ROOT / "data" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "model_analysis_escalation"

BEST_MODEL_PATH = MODELS_DIR / "best_escalation_72h_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics_escalation.json"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions_escalation.csv"

ANALYSIS_SUMMARY_PATH = OUTPUTS_DIR / "analysis_summary_escalation.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance_escalation.csv"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix_escalation.csv"


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run: python -m src.build_escalation_dataset"
        )

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "city"]).reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    excluded_cols = {
        "target_escalation_72h",
    }

    # Sve future/target kolone izbacujemo ako slučajno postoje
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded_cols
        and not col.startswith("target_")
        and not col.startswith("future_")
    ]

    X = df[feature_cols].copy()
    y = df["target_escalation_72h"].astype(int).copy()

    return X, y


def time_based_split(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    if "date" not in X.columns:
        raise ValueError("Expected 'date' column in feature set for time split.")

    unique_dates = sorted(pd.to_datetime(X["date"]).dt.normalize().unique())
    split_index = int(len(unique_dates) * (1 - test_ratio))
    split_index = max(1, min(split_index, len(unique_dates) - 1))

    split_date = unique_dates[split_index]

    train_mask = pd.to_datetime(X["date"]).dt.normalize() < split_date
    test_mask = pd.to_datetime(X["date"]).dt.normalize() >= split_date

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    return X_train, X_test, y_train, y_test, split_date


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    # raw date ne koristimo kao string feature; već imaš kalendarske featuree u datasetu
    columns_for_model = [c for c in X_train.columns if c != "date"]

    numeric_features = [
        c for c in columns_for_model if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical_features = [
        c for c in columns_for_model if c not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logistic_regression_escalation": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_escalation": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_escalation": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=450,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, pd.DataFrame]:
    X_test_model = X_test.drop(columns=["date"], errors="ignore")

    y_pred = model.predict(X_test_model)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_model)[:, 1]
    else:
        y_prob = np.zeros(len(y_test), dtype=float)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "precision_positive": round(float(precision), 4),
        "recall_positive": round(float(recall), 4),
        "f1_positive": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    meta_cols = [col for col in ["city", "date"] if col in X_test.columns]
    preds_df = X_test[meta_cols].copy() if meta_cols else pd.DataFrame(index=X_test.index)

    preds_df["actual_escalation_72h"] = y_test.values
    preds_df["predicted_escalation_72h"] = y_pred
    preds_df["predicted_escalation_probability"] = y_prob

    return metrics, preds_df


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def save_feature_importance(best_model: Pipeline) -> pd.DataFrame | None:
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    feature_names = get_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        if coef.ndim == 2:
            importance = np.abs(coef[0])
        else:
            importance = np.abs(coef)
    else:
        return None

    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    return fi_df


def save_confusion_matrix(best_metrics: dict) -> None:
    cm = best_metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test, split_date = time_based_split(X, y, test_ratio=0.2)

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    print("[INFO] Training escalation models...")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Split date: {pd.to_datetime(split_date).strftime('%Y-%m-%d')}")
    print()
    print("[INFO] Train target distribution:")
    print(y_train.value_counts().sort_index())
    print()
    print("[INFO] Test target distribution:")
    print(y_test.value_counts().sort_index())
    print()

    all_metrics: dict[str, dict] = {}
    all_predictions: list[pd.DataFrame] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"[INFO] Training: {model_name}")
        model.fit(X_train.drop(columns=["date"]), y_train)

        metrics, preds_df = evaluate_model(
            model,
            X_test,
            y_test,
        )

        preds_df["model_name"] = model_name

        all_metrics[model_name] = metrics
        all_predictions.append(preds_df)
        trained_models[model_name] = model

        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']}")
        print(f"Precision (pos): {metrics['precision_positive']}")
        print(f"Recall (pos): {metrics['recall_positive']}")
        print(f"F1 (pos): {metrics['f1_positive']}")
        print(f"ROC AUC: {metrics['roc_auc']}")
        print()

    # biramo po F1 pozitivne klase, pa onda ROC AUC
    def ranking_key(item):
        name, metrics = item
        return (
            metrics["f1_positive"],
            metrics["roc_auc"] if metrics["roc_auc"] is not None else -1,
            metrics["recall_positive"],
        )

    best_model_name, best_metrics = max(all_metrics.items(), key=ranking_key)
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, BEST_MODEL_PATH)

    metrics_payload = {
        "best_model": best_model_name,
        "split_date": pd.to_datetime(split_date).strftime("%Y-%m-%d"),
        "models": all_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    fi_df = save_feature_importance(best_model)
    save_confusion_matrix(best_metrics)

    analysis_summary = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split_date": pd.to_datetime(split_date).strftime("%Y-%m-%d"),
        "saved_best_model": best_model_name,
        "accuracy": best_metrics["accuracy"],
        "balanced_accuracy": best_metrics["balanced_accuracy"],
        "precision_positive": best_metrics["precision_positive"],
        "recall_positive": best_metrics["recall_positive"],
        "f1_positive": best_metrics["f1_positive"],
        "roc_auc": best_metrics["roc_auc"],
        "top_feature_count": int(len(fi_df.head(20))) if fi_df is not None else 0,
    }

    with open(ANALYSIS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

    print("[OK] Saved best escalation model to:")
    print(BEST_MODEL_PATH)
    print()
    print("[OK] Saved escalation metrics to:")
    print(METRICS_PATH)
    print()
    print("[OK] Saved escalation test predictions to:")
    print(PREDICTIONS_PATH)
    print()
    if fi_df is not None:
        print("[OK] Saved escalation feature importance to:")
        print(FEATURE_IMPORTANCE_PATH)
        print()
    print("[OK] Saved escalation analysis summary to:")
    print(ANALYSIS_SUMMARY_PATH)
    print()
    print("[OK] Best model selected:")
    print(best_model_name)
    print(f"Best F1 positive: {best_metrics['f1_positive']}")
    print(f"Best ROC AUC: {best_metrics['roc_auc']}")


if __name__ == "__main__":
    main()