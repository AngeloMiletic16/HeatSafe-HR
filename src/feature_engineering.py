from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


INPUT_PATH = PROCESSED_DATA_DIR / "all_cities_daily_with_risk.csv"
FEATURES_OUTPUT_PATH = PROCESSED_DATA_DIR / "all_cities_features.csv"
MODEL_DATASET_OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset_next_day.csv"

RISK_LEVEL_TO_CLASS = {
    "Nizak": 0,
    "Umjeren": 1,
    "Visok": 2,
    "Vrlo visok": 3,
}


def load_risk_dataset(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run risk engine first with: python -m src.risk_engine"
        )

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["month_sin"] = pd.Series(
        np.sin(2 * np.pi * df["month"] / 12), index=df.index
    )
    df["month_cos"] = pd.Series(
        np.cos(2 * np.pi * df["month"] / 12), index=df.index
    )
    df["day_of_year_sin"] = pd.Series(
        np.sin(2 * np.pi * df["day_of_year"] / 366), index=df.index
    )
    df["day_of_year_cos"] = pd.Series(
        np.cos(2 * np.pi * df["day_of_year"] / 366), index=df.index
    )

    return df


def add_base_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["hot_night_flag"] = (df["temp_min"] >= 20).astype(int)
    df["very_hot_day_flag"] = (df["apparent_temp_max"] >= 35).astype(int)
    df["dry_day_flag"] = (df["precipitation_sum"] < 1).astype(int)

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
        "heat_risk_score",
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
        "heat_risk_score",
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


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["target_next_day_score"] = df.groupby("city")["heat_risk_score"].shift(-1)
    df["target_next_day_risk_level"] = df.groupby("city")["risk_level"].shift(-1)
    df["target_next_day_class"] = df["target_next_day_risk_level"].map(RISK_LEVEL_TO_CLASS)

    return df


def finalize_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_columns] = df[numeric_columns].round(4)

    return df


def build_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep rows where lags and next-day target exist
    required_columns = [
        "temp_max_lag_1",
        "temp_max_lag_2",
        "temp_max_lag_3",
        "apparent_temp_max_lag_1",
        "humidity_mean_lag_1",
        "heat_risk_score_lag_1",
        "target_next_day_class",
    ]

    model_df = df.dropna(subset=required_columns).copy()
    model_df["target_next_day_class"] = model_df["target_next_day_class"].astype(int)

    return model_df


def save_outputs(features_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    features_df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    model_df.to_csv(MODEL_DATASET_OUTPUT_PATH, index=False)

    print("[OK] Saved full feature dataset to:")
    print(FEATURES_OUTPUT_PATH)
    print(f"Rows: {len(features_df):,}")
    print(f"Columns: {len(features_df.columns)}")

    print("\n[OK] Saved model-ready next-day dataset to:")
    print(MODEL_DATASET_OUTPUT_PATH)
    print(f"Rows: {len(model_df):,}")
    print(f"Columns: {len(model_df.columns)}")


def main() -> None:
    df = load_risk_dataset()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_base_derived_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_targets(df)
    df = finalize_feature_dataset(df)

    model_df = build_modeling_dataset(df)
    save_outputs(df, model_df)


if __name__ == "__main__":
    main()