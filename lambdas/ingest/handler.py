import csv
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "snowfall",
    "cloud_cover",
]
RENAME_MAP = {
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_gusts_10m": "wind_gusts_kmh",
    "precipitation": "precipitation_mm",
    "snowfall": "snowfall_cm",
    "cloud_cover": "cloud_cover_pct",
}


def load_airports(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def fetch_forecasts(airports: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    """Fetch next-48h hourly UTC forecasts for all hub airports in one Open-Meteo call."""
    params = {
        "latitude": ",".join(a["lat"] for a in airports),
        "longitude": ",".join(a["lon"] for a in airports),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_hours": 48,
        "timezone": "UTC",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data if isinstance(data, list) else [data]

    by_airport: dict[str, dict[str, dict[str, float]]] = {}
    for airport, item in zip(airports, items, strict=True):
        hourly = item["hourly"]
        times = hourly["time"]
        airport_hours: dict[str, dict[str, float]] = {}
        for i, t in enumerate(times):
            hour_key = f"{t}:00Z" if len(t) == 16 else f"{t}Z"
            airport_hours[hour_key] = {
                out_name: float(hourly[api_name][i])
                for api_name, out_name in RENAME_MAP.items()
            }
        by_airport[airport["code"]] = airport_hours
    return by_airport


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    bucket = os.environ["BUCKET_NAME"]
    csv_path = Path(os.environ.get("AIRPORTS_CSV", Path(__file__).parent / "airports.csv"))

    airports = load_airports(csv_path)
    forecasts = fetch_forecasts(airports)
    now = datetime.now(UTC)

    snapshot = {
        "updated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "airports": forecasts,
    }
    payload = json.dumps(snapshot)

    s3 = boto3.client("s3")
    raw_key = f"raw/weather/dt={now:%Y-%m-%d}/{now:%H%M}.json"
    s3.put_object(Bucket=bucket, Key=raw_key, Body=payload, ContentType="application/json")
    s3.put_object(
        Bucket=bucket,
        Key="features/latest_weather.json",
        Body=payload,
        ContentType="application/json",
    )

    logger.info(
        json.dumps(
            {
                "airports_updated": len(forecasts),
                "raw_key": raw_key,
                "updated_at_utc": snapshot["updated_at_utc"],
            }
        )
    )
    return {"statusCode": 200, "airports_updated": len(forecasts), "raw_key": raw_key}
