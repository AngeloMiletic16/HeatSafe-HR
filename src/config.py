from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

CITIES = {
    "Zagreb": {"lat": 45.8150, "lon": 15.9819},
    "Split": {"lat": 43.5081, "lon": 16.4402},
    "Rijeka": {"lat": 45.3271, "lon": 14.4422},
    "Osijek": {"lat": 45.5540, "lon": 18.6955},
    "Zadar": {"lat": 44.1194, "lon": 15.2314},
    "Dubrovnik": {"lat": 42.6507, "lon": 18.0944},
    "Šibenik": {"lat": 43.7350, "lon": 15.8952},
}

RISK_LABELS = {
    0: "Nizak",
    1: "Umjeren",
    2: "Visok",
    3: "Vrlo visok",
}

FORECAST_HORIZONS = [1, 3, 5]
DEFAULT_CITY = "Šibenik"

HISTORICAL_START_DATE = "2020-01-01"
HISTORICAL_END_DATE = date.today().isoformat()

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]