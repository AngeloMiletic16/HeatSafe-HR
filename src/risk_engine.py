from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


INPUT_PATH = PROCESSED_DATA_DIR / "all_cities_daily.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "all_cities_daily_with_risk.csv"


def load_daily_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run preprocessing first with: python -m src.preprocessing"
        )

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df["hot_day_32"] = (df["apparent_temp_max"] >= 32).astype(int)
    df["very_hot_day_35"] = (df["apparent_temp_max"] >= 35).astype(int)

    df["hot_days_last_3"] = (
        df.groupby("city")["hot_day_32"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )

    df["very_hot_days_last_3"] = (
        df.groupby("city")["very_hot_day_35"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )

    return df


def compute_apparent_temp_component(apparent_temp_max: pd.Series) -> pd.Series:
    score = np.select(
        [
            apparent_temp_max < 26,
            (apparent_temp_max >= 26) & (apparent_temp_max < 30),
            (apparent_temp_max >= 30) & (apparent_temp_max < 34),
            (apparent_temp_max >= 34) & (apparent_temp_max < 38),
            apparent_temp_max >= 38,
        ],
        [
            0,
            15,
            35,
            55,
            70,
        ],
        default=0,
    )
    return pd.Series(score, index=apparent_temp_max.index)


def compute_night_component(temp_min: pd.Series) -> pd.Series:
    score = np.select(
        [
            temp_min < 18,
            (temp_min >= 18) & (temp_min < 22),
            (temp_min >= 22) & (temp_min < 25),
            temp_min >= 25,
        ],
        [
            0,
            5,
            10,
            15,
        ],
        default=0,
    )
    return pd.Series(score, index=temp_min.index)


def compute_humidity_component(humidity_mean: pd.Series) -> pd.Series:
    score = np.select(
        [
            humidity_mean < 45,
            (humidity_mean >= 45) & (humidity_mean < 60),
            (humidity_mean >= 60) & (humidity_mean < 75),
            humidity_mean >= 75,
        ],
        [
            0,
            4,
            8,
            10,
        ],
        default=0,
    )
    return pd.Series(score, index=humidity_mean.index)


def compute_wind_component(wind_speed_mean: pd.Series) -> pd.Series:
    score = np.select(
        [
            wind_speed_mean >= 5,
            (wind_speed_mean >= 3) & (wind_speed_mean < 5),
            wind_speed_mean < 3,
        ],
        [
            0,
            3,
            6,
        ],
        default=0,
    )
    return pd.Series(score, index=wind_speed_mean.index)


def compute_persistence_component(
    hot_days_last_3: pd.Series,
    very_hot_days_last_3: pd.Series,
) -> pd.Series:
    score = (hot_days_last_3 * 3) + (very_hot_days_last_3 * 4)
    return score.clip(0, 14)


def compute_precipitation_adjustment(precipitation_sum: pd.Series) -> pd.Series:
    adjustment = np.select(
        [
            precipitation_sum >= 10,
            (precipitation_sum >= 3) & (precipitation_sum < 10),
            precipitation_sum < 3,
        ],
        [
            -6,
            -3,
            0,
        ],
        default=0,
    )
    return pd.Series(adjustment, index=precipitation_sum.index)


def assign_risk_label(score: pd.Series) -> pd.Series:
    labels = pd.cut(
        score,
        bins=[-1, 24, 49, 74, 100],
        labels=["Nizak", "Umjeren", "Visok", "Vrlo visok"],
    )
    return labels.astype(str)


def build_heat_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["score_apparent_temp"] = compute_apparent_temp_component(df["apparent_temp_max"])
    df["score_night"] = compute_night_component(df["temp_min"])
    df["score_humidity"] = compute_humidity_component(df["humidity_mean"])
    df["score_wind"] = compute_wind_component(df["wind_speed_mean"])
    df["score_persistence"] = compute_persistence_component(
        df["hot_days_last_3"],
        df["very_hot_days_last_3"],
    )
    df["score_precip_adjustment"] = compute_precipitation_adjustment(df["precipitation_sum"])

    df["heat_risk_score"] = (
        df["score_apparent_temp"]
        + df["score_night"]
        + df["score_humidity"]
        + df["score_wind"]
        + df["score_persistence"]
        + df["score_precip_adjustment"]
    ).clip(0, 100)

    df["risk_level"] = assign_risk_label(df["heat_risk_score"])

    return df


def save_outputs(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    df.to_csv(output_path, index=False)
    print("[OK] Saved risk-enriched dataset to:")
    print(output_path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


def main() -> None:
    df = load_daily_data()
    df = add_persistence_features(df)
    df = build_heat_risk_score(df)
    save_outputs(df)


if __name__ == "__main__":
    main()