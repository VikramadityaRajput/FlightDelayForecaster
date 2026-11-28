import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fdf.features import FEATURE_COLUMNS, add_temporal_features, apply_priors, chronological_split

ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = ROOT / "lambdas" / "predict" / "handler.py"
DATA_PATH = ROOT / "data" / "interim" / "flights_with_weather.parquet"
AIRPORTS_CSV = ROOT / "config" / "airports.csv"


def _load_handler():
    spec = importlib.util.spec_from_file_location("predict_handler", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


handler = _load_handler()

WEATHER_FIELDS = [
    f"{prefix}_{name}"
    for prefix in ("origin", "dest")
    for name in (
        "temp_c",
        "wind_speed_kmh",
        "wind_gusts_kmh",
        "precipitation_mm",
        "snowfall_cm",
        "cloud_cover_pct",
    )
]


def test_handler_matches_offline_pipeline_on_20_test_rows():
    models_dir = max(p for p in (ROOT / "models").iterdir() if p.is_dir())
    bundle = joblib.load(models_dir / "model.joblib")
    metadata = json.loads((models_dir / "feature_metadata.json").read_text())

    df = pd.read_parquet(DATA_PATH).dropna(subset=["scheduled_dep_utc"])
    _, _, test = chronological_split(df)
    sample = test.iloc[:20].copy()

    offline_df = add_temporal_features(sample, AIRPORTS_CSV)
    offline_df = apply_priors(offline_df, metadata["priors"])
    X = offline_df[FEATURE_COLUMNS]
    offline_proba = bundle["classifier"].predict_proba(X)[:, 1]
    buckets = metadata["severity_buckets"]
    offline_sev = bundle["severity_classifier"].predict_proba(X)
    offline_p50 = np.clip(np.expm1(bundle["regressor_p50"].predict(X)), 15.0, None)
    offline_p90 = np.maximum(offline_p50, np.expm1(bundle["regressor_p90"].predict(X)))

    for i, (_, row) in enumerate(sample.iterrows()):
        payload = {
            "origin_airport": row["Origin"],
            "dest_airport": row["Dest"],
            "carrier": row["Reporting_Airline"],
            "scheduled_dep_utc": row["scheduled_dep_utc"].isoformat(),
            "scheduled_duration_min": float(row["CRSElapsedTime"]),
            "distance_miles": float(row["Distance"]),
            "weather_override": {col: float(row[col]) for col in WEATHER_FIELDS},
        }
        resp = handler.lambda_handler({"body": json.dumps(payload)})
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["weather_source"] == "override"
        assert body["delay_probability"] == round(float(offline_proba[i]), 4)
        expected_delayed = bool(offline_proba[i] >= metadata["threshold"])
        expected_sev = buckets[int(np.argmax(offline_sev[i]))]
        assert body["is_delayed"] == expected_delayed
        assert body["severity_if_delayed"] == expected_sev
        assert body["predicted_bucket"] == (expected_sev if expected_delayed else "on_time")
        assert body["delay_if_delayed_min"] == round(float(offline_p50[i]), 1)
        assert body["delay_if_delayed_p90_min"] == round(float(offline_p90[i]), 1)
        assert body["estimated_delay_min"] == round(float(offline_proba[i] * offline_p50[i]), 2)
        assert body["bucket_probabilities"]["on_time"] == round(float(1.0 - offline_proba[i]), 4)
        for label, sp in zip(buckets, offline_sev[i], strict=True):
            assert body["bucket_probabilities"][label] == round(float(offline_proba[i] * sp), 4)


def test_forecast_weather_lookup_and_partial_override(tmp_path, monkeypatch):
    dep = (datetime.now(UTC) + timedelta(hours=3)).replace(minute=30, second=0, microsecond=0)
    dep_hour = dep.replace(minute=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    arr_hour = (dep + timedelta(minutes=120)).replace(minute=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    base_weather = {
        "temp_c": 15.0,
        "wind_speed_kmh": 10.0,
        "wind_gusts_kmh": 18.0,
        "precipitation_mm": 0.0,
        "snowfall_cm": 0.0,
        "cloud_cover_pct": 40.0,
    }
    weather_file = tmp_path / "latest_weather.json"
    weather_file.write_text(
        json.dumps(
            {
                "airports": {
                    "ORD": {dep_hour: base_weather},
                    "EWR": {arr_hour: base_weather},
                }
            }
        )
    )
    monkeypatch.setenv("WEATHER_PATH", str(weather_file))

    payload = {
        "origin_airport": "ORD",
        "dest_airport": "EWR",
        "carrier": "UA",
        "scheduled_dep_utc": dep.isoformat(),
        "scheduled_duration_min": 120,
        "distance_miles": 719,
        "weather_override": {"origin_wind_speed_kmh": 30.0},
    }
    resp = handler.lambda_handler({"body": json.dumps(payload)})
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["weather_source"] == "forecast+override"


@pytest.mark.parametrize(
    "event",
    [
        {"body": "{not-json"},
        {"body": json.dumps({"origin_airport": "ORD"})},
        {
            "body": json.dumps(
                {
                    "origin_airport": "ORD",
                    "dest_airport": "EWR",
                    "carrier": "UA",
                    "scheduled_dep_utc": "not-a-timestamp",
                    "scheduled_duration_min": 120,
                    "distance_miles": 719,
                }
            )
        },
        {
            "body": json.dumps(
                {
                    "origin_airport": "ORD",
                    "dest_airport": "EWR",
                    "carrier": "UA",
                    "scheduled_dep_utc": "2026-11-25T18:30:00Z",
                    "scheduled_duration_min": -10,
                    "distance_miles": 719,
                }
            )
        },
    ],
)
def test_bad_inputs_return_400(event):
    resp = handler.lambda_handler(event)
    assert resp["statusCode"] == 400
    assert "error" in json.loads(resp["body"])


def test_unknown_hub_returns_422():
    payload = {
        "origin_airport": "ZZZ",
        "dest_airport": "EWR",
        "carrier": "UA",
        "scheduled_dep_utc": "2026-11-25T18:30:00Z",
        "scheduled_duration_min": 120,
        "distance_miles": 719,
    }
    resp = handler.lambda_handler({"body": json.dumps(payload)})
    assert resp["statusCode"] == 422


def test_outside_forecast_window_without_override_returns_422():
    far_future = (datetime.now(UTC) + timedelta(hours=72)).isoformat()
    payload = {
        "origin_airport": "ORD",
        "dest_airport": "EWR",
        "carrier": "UA",
        "scheduled_dep_utc": far_future,
        "scheduled_duration_min": 120,
        "distance_miles": 719,
    }
    resp = handler.lambda_handler({"body": json.dumps(payload)})
    assert resp["statusCode"] == 422
