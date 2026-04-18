import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


INPUT_PATH = PROCESSED_DATA_DIR / "model_dataset_next_day.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_next_day_risk_model_strict.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics_strict.json"
TEST_PREDICTIONS_PATH = MODELS_DIR / "test_predictions_strict.csv"

TARGET_COLUMN = "target_next_day_class"
TEST_DAYS = 365

DROP_COLUMNS = [
    "date",
    "risk_level",
    "heat_risk_score",
    "target_next_day_score",
    "target_next_day_risk_level",
    "target_next_day_class",
    "score_apparent_temp",
    "score_night",
    "score_humidity",
    "score_wind",
    "score_persistence",
    "score_precip_adjustment",
]

CATEGORICAL_FEATURES = [
    "city",
]

NUMERIC_EXCLUDE = set(DROP_COLUMNS + CATEGORICAL_FEATURES)


def load_dataset(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run feature engineering first with: python -m src.feature_engineering"
        )

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    forbidden_prefixes = ("target_", "score_")
    forbidden_contains = ("heat_risk_score",)

    numeric_features = []
    for col in df.columns:
        if col in NUMERIC_EXCLUDE:
            continue
        if col.startswith(forbidden_prefixes):
            continue
        if any(token in col for token in forbidden_contains):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)

    categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    return numeric_features, categorical_features


def time_based_split(df: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Adjust TEST_DAYS.")

    return train_df, test_df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression_strict": logistic_model,
        "random_forest_strict": random_forest_model,
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, pd.Series]:
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    return metrics, pd.Series(predictions, index=x_test.index, name="prediction")


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_outputs(best_model: Pipeline, all_metrics: dict, test_results_df: pd.DataFrame) -> None:
    ensure_models_dir()

    joblib.dump(best_model, BEST_MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    test_results_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    print("[OK] Saved strict best model to:")
    print(BEST_MODEL_PATH)
    print("\n[OK] Saved strict metrics to:")
    print(METRICS_PATH)
    print("\n[OK] Saved strict test predictions to:")
    print(TEST_PREDICTIONS_PATH)


def main() -> None:
    df = load_dataset()
    numeric_features, categorical_features = build_feature_lists(df)

    train_df, test_df = time_based_split(df, test_days=TEST_DAYS)

    x_train = train_df[numeric_features + categorical_features]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[numeric_features + categorical_features]
    y_test = test_df[TARGET_COLUMN]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    all_metrics = {}
    best_model_name = None
    best_model = None
    best_macro_f1 = -1
    best_predictions = None

    print("[INFO] Training STRICT models...")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    print("\n[INFO] Train target distribution:")
    print(train_df[TARGET_COLUMN].value_counts().sort_index())

    print("\n[INFO] Test target distribution:")
    print(test_df[TARGET_COLUMN].value_counts().sort_index())

    for model_name, model in models.items():
        print(f"\n[INFO] Training: {model_name}")
        model.fit(x_train, y_train)

        metrics, predictions = evaluate_model(model, x_test, y_test)
        all_metrics[model_name] = metrics

        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Macro F1: {metrics['macro_f1']}")
        print(f"Weighted F1: {metrics['weighted_f1']}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_model_name = model_name
            best_model = model
            best_predictions = predictions

    all_metrics["best_model"] = best_model_name

    test_results_df = test_df[["city", "date", "risk_level", "heat_risk_score", TARGET_COLUMN]].copy()
    test_results_df["prediction"] = best_predictions.values

    save_outputs(best_model, all_metrics, test_results_df)

    print("\n[OK] Best STRICT model selected:")
    print(best_model_name)
    print(f"Best STRICT Macro F1: {best_macro_f1}")


if __name__ == "__main__":
    main()