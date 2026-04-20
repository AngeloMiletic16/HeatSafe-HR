from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "models" / "test_predictions_escalation.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_analysis_escalation"
THRESHOLD_RESULTS_PATH = OUTPUT_DIR / "threshold_tuning_escalation.csv"
THRESHOLD_SUMMARY_PATH = OUTPUT_DIR / "threshold_tuning_summary.json"


def evaluate_threshold(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_positive": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_positive": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_positive": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


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
    y_prob = best_model_df["predicted_escalation_probability"].astype(float)

    thresholds = [round(x, 2) for x in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]]
    results = [evaluate_threshold(y_true, y_prob, t) for t in thresholds]

    results_df = pd.DataFrame(results).sort_values("threshold").reset_index(drop=True)
    results_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)

    # Fokus: dobar recall + dobar precision + stabilan F1
    best_by_f1 = results_df.sort_values(
        ["f1_positive", "recall_positive", "precision_positive"],
        ascending=[False, False, False],
    ).iloc[0]

    # Prijedlog label pragova
    suggested_thresholds = {
        "stable_threshold_upper": 0.30,
        "watch_threshold_upper": 0.60,
        "likely_escalation_lower": 0.60,
    }

    summary = {
        "evaluated_thresholds": thresholds,
        "best_by_f1": best_by_f1.to_dict(),
        "suggested_thresholds": suggested_thresholds,
        "notes": [
            "Stable: probability < 0.30",
            "Watch: 0.30 <= probability < 0.60",
            "Likely escalation: probability >= 0.60",
            "These thresholds can be tightened later based on false positives vs false negatives tradeoff.",
        ],
    }

    with open(THRESHOLD_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Saved threshold tuning table to:")
    print(THRESHOLD_RESULTS_PATH)
    print()
    print("[OK] Saved threshold tuning summary to:")
    print(THRESHOLD_SUMMARY_PATH)
    print()
    print("[INFO] Best threshold by F1:")
    print(best_by_f1.to_dict())


if __name__ == "__main__":
    main()