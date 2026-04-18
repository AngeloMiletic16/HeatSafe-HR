from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

INPUT_PATH = PROCESSED_DATA_DIR / "model_dataset_next_day.csv"
MODEL_PATH = MODELS_DIR / "best_next_day_risk_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

TARGET_COLUMN = "target_next_day_class"
TEST_DAYS = 365

CLASS_ID_TO_LABEL = {
    0: "Nizak",
    1: "Umjeren",
    2: "Visok",
    3: "Vrlo visok",
}


def get_output_dir() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "model_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. "
            "Run feature engineering first with: python -m src.feature_engineering"
        )

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def time_based_split(df: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Adjust TEST_DAYS.")

    return train_df, test_df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run training first with: python -m src.train_model"
        )
    return joblib.load(MODEL_PATH)


def build_x_test(model, test_df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)
        return test_df[feature_columns].copy()

    raise ValueError("Model does not expose feature_names_in_. Cannot reconstruct test features safely.")


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_dir: Path) -> None:
    labels = [0, 1, 2, 3]
    class_names = [CLASS_ID_TO_LABEL[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    cm_png_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved confusion matrix CSV to:")
    print(cm_csv_path)
    print("[OK] Saved confusion matrix PNG to:")
    print(cm_png_path)


def save_classification_report(y_true: pd.Series, y_pred: pd.Series, output_dir: Path) -> None:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    # Add readable labels for class rows
    report_df = report_df.rename(
        index={
            "0": "Nizak",
            "1": "Umjeren",
            "2": "Visok",
            "3": "Vrlo visok",
        }
    )

    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)

    print("[OK] Saved classification report to:")
    print(report_path)


def save_predictions(test_df: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, model, x_test: pd.DataFrame, output_dir: Path) -> None:
    results_df = test_df[["city", "date", "risk_level", "heat_risk_score"]].copy()
    results_df["true_class"] = y_true.values
    results_df["predicted_class"] = y_pred.values
    results_df["true_label"] = results_df["true_class"].map(CLASS_ID_TO_LABEL)
    results_df["predicted_label"] = results_df["predicted_class"].map(CLASS_ID_TO_LABEL)
    results_df["correct_prediction"] = (results_df["true_class"] == results_df["predicted_class"]).astype(int)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        for idx, class_id in enumerate(model.classes_):
            results_df[f"proba_class_{class_id}"] = proba[:, idx]

    predictions_path = output_dir / "test_predictions_detailed.csv"
    results_df.to_csv(predictions_path, index=False)

    print("[OK] Saved detailed test predictions to:")
    print(predictions_path)


def extract_feature_importance(model) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_).mean(axis=0)
    else:
        raise ValueError("This classifier does not expose feature importances or coefficients.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def save_feature_importance(importance_df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    plot_path = output_dir / "feature_importance_top20.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved full feature importance table to:")
    print(importance_path)
    print("[OK] Saved feature importance chart to:")
    print(plot_path)


def save_summary_json(y_true: pd.Series, y_pred: pd.Series, output_dir: Path) -> None:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary = {
        "test_rows": int(len(y_true)),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            training_metrics = json.load(f)
        summary["saved_best_model"] = training_metrics.get("best_model")

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Saved analysis summary to:")
    print(summary_path)
    print("\n[INFO] Analysis summary:")
    print(summary)


def main() -> None:
    output_dir = get_output_dir()

    df = load_dataset()
    _, test_df = time_based_split(df, test_days=TEST_DAYS)

    model = load_model()
    x_test = build_x_test(model, test_df)
    y_true = test_df[TARGET_COLUMN].copy()
    y_pred = pd.Series(model.predict(x_test), index=test_df.index)

    save_confusion_matrix(y_true, y_pred, output_dir)
    save_classification_report(y_true, y_pred, output_dir)
    save_predictions(test_df, y_true, y_pred, model, x_test, output_dir)

    importance_df = extract_feature_importance(model)
    save_feature_importance(importance_df, output_dir, top_n=20)

    save_summary_json(y_true, y_pred, output_dir)

    print("\n[OK] Model analysis finished successfully.")


if __name__ == "__main__":
    main()