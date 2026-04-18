import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    CITIES,
    HISTORICAL_END_DATE,
    HISTORICAL_START_DATE,
    HOURLY_VARIABLES,
    RAW_DATA_DIR,
)

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def slugify_city_name(city_name: str) -> str:
    """
    Convert city names like 'Šibenik' into a safe filename like 'sibenik'.
    """
    normalized = unicodedata.normalize("NFKD", city_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_")


def ensure_raw_directories() -> Path:
    """
    Ensure the raw data folder exists and create a dedicated open_meteo subfolder.
    """
    raw_open_meteo_dir = RAW_DATA_DIR / "open_meteo"
    raw_open_meteo_dir.mkdir(parents=True, exist_ok=True)
    return raw_open_meteo_dir


def build_archive_params(lat: float, lon: float) -> dict[str, Any]:
    """
    Build API parameters for Open-Meteo historical hourly data.
    """
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": HISTORICAL_START_DATE,
        "end_date": HISTORICAL_END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",
    }


def fetch_city_historical_data(city_name: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch hourly historical weather data for a single city from Open-Meteo.
    """
    params = build_archive_params(lat=lat, lon=lon)

    response = requests.get(ARCHIVE_API_URL, params=params, timeout=60)
    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError(f"No hourly data returned for city '{city_name}'.")

    hourly_data = payload["hourly"]
    df = pd.DataFrame(hourly_data)

    # Convert timestamp column
    df["time"] = pd.to_datetime(df["time"])

    # Add metadata
    df["city"] = city_name
    df["latitude"] = lat
    df["longitude"] = lon

    # Reorder columns
    ordered_columns = ["city", "latitude", "longitude", "time"] + HOURLY_VARIABLES
    df = df[ordered_columns]

    return df


def save_city_dataframe(df: pd.DataFrame, city_name: str, output_dir: Path) -> Path:
    """
    Save a city's hourly dataframe to CSV.
    """
    city_slug = slugify_city_name(city_name)
    output_path = output_dir / f"{city_slug}_historical_hourly.csv"
    df.to_csv(output_path, index=False)
    return output_path


def fetch_and_save_one_city(city_name: str) -> Path:
    """
    Fetch and save historical hourly data for a single city defined in config.
    """
    if city_name not in CITIES:
        valid_cities = ", ".join(CITIES.keys())
        raise ValueError(f"Unknown city '{city_name}'. Valid cities: {valid_cities}")

    output_dir = ensure_raw_directories()
    city_info = CITIES[city_name]

    df = fetch_city_historical_data(
        city_name=city_name,
        lat=city_info["lat"],
        lon=city_info["lon"],
    )

    output_path = save_city_dataframe(df=df, city_name=city_name, output_dir=output_dir)

    print(f"[OK] Saved {city_name} historical hourly data to:")
    print(output_path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return output_path


def fetch_and_save_all_cities() -> Path:
    """
    Fetch and save historical hourly data for all configured cities,
    plus a merged CSV for convenience.
    """
    output_dir = ensure_raw_directories()
    all_dataframes: list[pd.DataFrame] = []

    for city_name, city_info in CITIES.items():
        print(f"[INFO] Fetching historical data for {city_name}...")
        df = fetch_city_historical_data(
            city_name=city_name,
            lat=city_info["lat"],
            lon=city_info["lon"],
        )
        save_city_dataframe(df=df, city_name=city_name, output_dir=output_dir)
        all_dataframes.append(df)

    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_output_path = output_dir / "all_cities_historical_hourly.csv"
    merged_df.to_csv(merged_output_path, index=False)

    print("[OK] Saved merged historical hourly data to:")
    print(merged_output_path)
    print(f"Rows: {len(merged_df):,}")
    print(f"Columns: {len(merged_df.columns)}")

    return merged_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical hourly weather data for HeatSafe HR."
    )
    parser.add_argument(
        "--city",
        type=str,
        help="Fetch data for a single city, e.g. --city 'Šibenik'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch data for all configured cities",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.all:
            fetch_and_save_all_cities()
        elif args.city:
            fetch_and_save_one_city(args.city)
        else:
            print("No arguments provided.")
            print("Use one of the following:")
            print("  python -m src.data_ingestion --city \"Šibenik\"")
            print("  python -m src.data_ingestion --all")
            sys.exit(1)

    except requests.HTTPError as exc:
        print(f"[ERROR] HTTP error while calling Open-Meteo: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()