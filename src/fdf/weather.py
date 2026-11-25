from pathlib import Path
import time
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "snowfall",
    "cloud_cover",
]
RENAME_MAP = {
    "time": "time_utc",
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_gusts_10m": "wind_gusts_kmh",
    "precipitation": "precipitation_mm",
    "snowfall": "snowfall_cm",
    "cloud_cover": "cloud_cover_pct",
}


def fetch_airport_weather(
    code: str, lat: float, lon: float, start_date: str, end_date: str, cache_dir: Path
) -> pd.DataFrame:
    """Fetch hourly UTC weather for one airport from Open-Meteo, caching to Parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{code}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json()["hourly"]).rename(columns=RENAME_MAP)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["airport"] = code
    df.to_parquet(cache_path, index=False)
    time.sleep(0.3)
    return df


def load_all_weather(
    airports_csv: Path, start_date: str, end_date: str, cache_dir: Path
) -> pd.DataFrame:
    """Load hourly UTC weather for every airport in airports_csv."""
    airports = pd.read_csv(airports_csv)
    frames = [
        fetch_airport_weather(row.code, row.lat, row.lon, start_date, end_date, cache_dir)
        for row in airports.itertuples(index=False)
    ]
    return pd.concat(frames, ignore_index=True)