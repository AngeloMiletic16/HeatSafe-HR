from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "models" / "test_predictions_escalation.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "model_analysis_escalation" / "feature_importance_escalation.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_analysis_escalation"

CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report_escalation.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_escalation_detailed.csv"
FALSE_POSITIVES_PATH = OUTPUT_DIR / "false_positives_escalation.csv"
FALSE_NEGATIVES_PATH = OUTPUT_DIR / "false_negatives_escalation.csv"
ANALYSIS_SUMMARY_PATH = OUTPUT_DIR / "analysis_summary_escalation_detailed.json"


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}\n"
            "Run: python -m src.train_escalation_model"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_PATH)
    best_model_df = df[df["model_name"] == "extra_trees_escalation"].copy()

    if best_model_df.empty:
        raise ValueError("No rows found for best model 'extra_trees_escalation'.")

    y_true = best_model_df["actual_escalation_72h"].astype(int)
    y_pred = best_model_df["predicted_escalation_72h"].astype(int)
    y_prob = best_model_df["predicted_escalation_probability"].astype(float)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        [[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    fp_df = best_model_df[(best_model_df["actual_escalation_72h"] == 0) & (best_model_df["predicted_escalation_72h"] == 1)].copy()
    fn_df = best_model_df[(best_model_df["actual_escalation_72h"] == 1) & (best_model_df["predicted_escalation_72h"] == 0)].copy()

    fp_df.to_csv(FALSE_POSITIVES_PATH, index=False)
    fn_df.to_csv(FALSE_NEGATIVES_PATH, index=False)

    roc_auc = roc_auc_score(y_true, y_prob)

    top_features = []
    if FEATURE_IMPORTANCE_PATH.exists():
        fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        top_features = fi_df.head(15).to_dict(orient="records")

    summary = {
        "model_name": "extra_trees_escalation",
        "test_rows": int(len(best_model_df)),
        "roc_auc": round(float(roc_auc), 4),
        "positive_class_precision": round(float(report["1"]["precision"]), 4),
        "positive_class_recall": round(float(report["1"]["recall"]), 4),
        "positive_class_f1": round(float(report["1"]["f1-score"]), 4),
        "false_positives": int(len(fp_df)),
        "false_negatives": int(len(fn_df)),
        "top_features": top_features,
    }

    with open(ANALYSIS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Saved classification report to:")
    print(CLASSIFICATION_REPORT_PATH)
    print()
    print("[OK] Saved detailed confusion matrix to:")
    print(CONFUSION_MATRIX_PATH)
    print()
    print("[OK] Saved false positives to:")
    print(FALSE_POSITIVES_PATH)
    print()
    print("[OK] Saved false negatives to:")
    print(FALSE_NEGATIVES_PATH)
    print()
    print("[OK] Saved detailed escalation summary to:")
    print(ANALYSIS_SUMMARY_PATH)


if __name__ == "__main__":
    main()