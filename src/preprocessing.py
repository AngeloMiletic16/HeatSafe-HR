from pathlib import Path
import unicodedata

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


RAW_INPUT_PATH = RAW_DATA_DIR / "open_meteo" / "all_cities_historical_hourly.csv"


def slugify_city_name(city_name: str) -> str:
    """
    Convert city names like 'Šibenik' into a safe filename like 'sibenik'.
    """
    normalized = unicodedata.normalize("NFKD", city_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_")


def ensure_processed_directories() -> Path:
    """
    Ensure the processed data directory exists.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_DIR


def load_hourly_data(input_path: Path = RAW_INPUT_PATH) -> pd.DataFrame:
    """
    Load merged hourly weather data from CSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run data ingestion first with: python -m src.data_ingestion --all"
        )

    df = pd.read_csv(input_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that the expected columns exist.
    """
    required_columns = [
        "city",
        "latitude",
        "longitude",
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - parse datetime
    - sort rows
    - remove duplicate rows if any
    """
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    numeric_columns = [
        "latitude",
        "longitude",
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    return df


def aggregate_hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly weather data into a daily city-level dataset.
    """
    df = df.copy()
    df["date"] = df["time"].dt.floor("D")

    daily_df = (
        df.groupby(["city", "latitude", "longitude", "date"], as_index=False)
        .agg(
            temp_mean=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            humidity_mean=("relative_humidity_2m", "mean"),
            humidity_max=("relative_humidity_2m", "max"),
            apparent_temp_mean=("apparent_temperature", "mean"),
            apparent_temp_max=("apparent_temperature", "max"),
            precipitation_sum=("precipitation", "sum"),
            pressure_mean=("surface_pressure", "mean"),
            wind_speed_mean=("wind_speed_10m", "mean"),
            wind_speed_max=("wind_speed_10m", "max"),
            hourly_records=("time", "count"),
        )
        .sort_values(["city", "date"])
        .reset_index(drop=True)
    )

    # Optional rounding for cleaner output
    rounded_columns = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "humidity_mean",
        "humidity_max",
        "apparent_temp_mean",
        "apparent_temp_max",
        "precipitation_sum",
        "pressure_mean",
        "wind_speed_mean",
        "wind_speed_max",
    ]

    daily_df[rounded_columns] = daily_df[rounded_columns].round(2)

    return daily_df


def save_daily_outputs(daily_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save merged daily dataset and one CSV per city.
    """
    merged_output_path = output_dir / "all_cities_daily.csv"
    daily_df.to_csv(merged_output_path, index=False)

    for city_name in daily_df["city"].unique():
        city_df = daily_df[daily_df["city"] == city_name].copy()
        city_slug = slugify_city_name(city_name)
        city_output_path = output_dir / f"{city_slug}_daily.csv"
        city_df.to_csv(city_output_path, index=False)

    print("[OK] Saved merged daily dataset to:")
    print(merged_output_path)
    print(f"Rows: {len(daily_df):,}")
    print(f"Columns: {len(daily_df.columns)}")


def main() -> None:
    output_dir = ensure_processed_directories()

    hourly_df = load_hourly_data()
    validate_columns(hourly_df)
    hourly_df = clean_hourly_data(hourly_df)

    daily_df = aggregate_hourly_to_daily(hourly_df)
    save_daily_outputs(daily_df, output_dir)


if __name__ == "__main__":
    main()