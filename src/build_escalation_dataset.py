from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset_escalation_72h.csv"

SEVERE_LEVELS = {"Visok", "Vrlo visok"}


def build_escalation_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run the feature engineering pipeline first so all_cities_features.csv exists."
        )

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    original_columns = df.columns.tolist()

    grouped = df.groupby("city", group_keys=False)

    df["future_risk_level_d1"] = grouped["risk_level"].shift(-1)
    df["future_risk_level_d2"] = grouped["risk_level"].shift(-2)
    df["future_risk_level_d3"] = grouped["risk_level"].shift(-3)

    # Tražimo puni 72h prozor.
    # Ako nema t+3 dana, red ne želimo u train datasetu.
    valid_mask = df["future_risk_level_d3"].notna()

    df["target_escalation_72h"] = (
        df["future_risk_level_d1"].isin(SEVERE_LEVELS)
        | df["future_risk_level_d2"].isin(SEVERE_LEVELS)
        | df["future_risk_level_d3"].isin(SEVERE_LEVELS)
    ).astype(int)

    output_df = df.loc[valid_mask, original_columns + ["target_escalation_72h"]].copy()

    output_df.to_csv(OUTPUT_PATH, index=False)

    return output_df


def main() -> None:
    df_out = build_escalation_dataset()

    print("[OK] Saved escalation dataset to:")
    print(OUTPUT_PATH)
    print(f"Rows: {len(df_out):,}")
    print(f"Columns: {df_out.shape[1]}")
    print()
    print("[INFO] target_escalation_72h distribution:")
    print(df_out["target_escalation_72h"].value_counts().sort_index())


if __name__ == "__main__":
    main()