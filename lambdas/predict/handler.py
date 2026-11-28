import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import joblib
import numpy as np
import pandas as pd

from fdf.features import FEATURE_COLUMNS, add_temporal_features, apply_priors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REQUIRED_FIELDS = (
    "origin_airport",
    "dest_airport",
    "carrier",
    "scheduled_dep_utc",
    "scheduled_duration_min",
    "distance_miles",
)

WEATHER_BASE_FIELDS = (
    "temp_c",
    "wind_speed_kmh",
    "wind_gusts_kmh",
    "precipitation_mm",
    "snowfall_cm",
    "cloud_cover_pct",
)

ALL_WEATHER_FIELDS = tuple(
    f"{prefix}_{field}" for prefix in ("origin", "dest") for field in WEATHER_BASE_FIELDS
)

_CACHE: dict[str, Any] = {}


def _default_model_dir() -> Path:
    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    models_root = Path(__file__).resolve().parents[2] / "models"
    versions = sorted(p for p in models_root.iterdir() if p.is_dir())
    if not versions:
        raise RuntimeError(f"No model directories found under {models_root}")
    return versions[-1]


def _default_airports_csv() -> Path:
    env_csv = os.environ.get("AIRPORTS_CSV")
    if env_csv:
        return Path(env_csv)
    return Path(__file__).resolve().parents[2] / "config" / "airports.csv"


def _download_s3_prefix(bucket: str, prefix: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    for name in ("model.joblib", "feature_metadata.json"):
        s3.download_file(bucket, f"{prefix}/{name}", str(dest_dir / name))


def load_artifacts(force_reload: bool = False) -> dict[str, Any]:
    """Load model pipeline, metadata, and airports table once per container."""
    if _CACHE and not force_reload:
        return _CACHE

    bucket = os.environ.get("BUCKET_NAME")
    model_version = os.environ.get("MODEL_VERSION")

    if bucket and model_version and not os.environ.get("MODEL_DIR"):
        model_dir = Path("/tmp/model")
        if not (model_dir / "model.joblib").exists():
            _download_s3_prefix(bucket, f"models/{model_version}", model_dir)
    else:
        model_dir = _default_model_dir()

    bundle = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "feature_metadata.json").read_text())
    airports_csv = _default_airports_csv()

    _CACHE.clear()
    _CACHE.update(
        {
            "classifier": bundle["classifier"],
            "severity_classifier": bundle.get("severity_classifier"),
            "regressor_p50": bundle.get("regressor_p50"),
            "regressor_p90": bundle.get("regressor_p90"),
            "regressor": bundle.get("regressor"),
            "metadata": metadata,
            "airports_csv": airports_csv,
            "hubs": set(metadata["hubs"]),
        }
    )
    return _CACHE


def _load_latest_weather() -> dict[str, Any] | None:
    """Load latest_weather.json from local WEATHER_PATH or S3 BUCKET_NAME."""
    weather_path = os.environ.get("WEATHER_PATH")
    if weather_path:
        p = Path(weather_path)
        return json.loads(p.read_text()) if p.exists() else None

    bucket = os.environ.get("BUCKET_NAME")
    if bucket:
        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key="features/latest_weather.json")
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load latest_weather.json from S3: %s", exc)
            return None
    return None


def _response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def _parse_payload(event: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if "body" in event:
        raw = event["body"]
        if raw is None:
            return None, "Request body is required."
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None, "Invalid JSON in request body."
        elif isinstance(raw, dict):
            parsed = raw
        else:
            return None, "Unsupported request body type."
    else:
        parsed = event

    if not isinstance(parsed, dict):
        return None, "Request payload must be a JSON object."
    return parsed, None


def _validate_and_parse_request(
    payload: dict[str, Any], hubs: set[str]
) -> tuple[dict[str, Any] | None, tuple[int, str] | None]:
    for field in REQUIRED_FIELDS:
        if field not in payload or payload[field] is None or payload[field] == "":
            return None, (400, f"Missing required field: {field}")

    origin = str(payload["origin_airport"]).strip().upper()
    dest = str(payload["dest_airport"]).strip().upper()
    carrier = str(payload["carrier"]).strip().upper()

    try:
        duration_min = float(payload["scheduled_duration_min"])
        distance_miles = float(payload["distance_miles"])
    except (TypeError, ValueError):
        return None, (400, "scheduled_duration_min and distance_miles must be numbers.")

    if duration_min <= 0 or distance_miles <= 0:
        return None, (400, "scheduled_duration_min and distance_miles must be positive.")

    try:
        dep_ts = pd.Timestamp(payload["scheduled_dep_utc"])
        if pd.isna(dep_ts):
            raise ValueError("NaT")
        if dep_ts.tzinfo is None:
            dep_ts = dep_ts.tz_localize("UTC")
        else:
            dep_ts = dep_ts.tz_convert("UTC")
    except Exception:  # noqa: BLE001
        return None, (400, "Unparseable scheduled_dep_utc timestamp.")

    override = payload.get("weather_override") or {}
    if not isinstance(override, dict):
        return None, (400, "weather_override must be a JSON object.")

    if origin not in hubs or dest not in hubs:
        return None, (422, f"Unsupported airport(s): origin={origin}, dest={dest}")

    return {
        "origin_airport": origin,
        "dest_airport": dest,
        "carrier": carrier,
        "scheduled_dep_utc": dep_ts,
        "scheduled_duration_min": duration_min,
        "distance_miles": distance_miles,
        "weather_override": override,
    }, None


def _resolve_weather(
    req: dict[str, Any],
) -> tuple[dict[str, float] | None, str | None, list[str], tuple[int, str] | None]:
    override = req["weather_override"]
    warnings: list[str] = []
    weather_values: dict[str, float] = {}

    for field in ALL_WEATHER_FIELDS:
        if field in override and override[field] is not None:
            try:
                weather_values[field] = float(override[field])
            except (TypeError, ValueError):
                return None, None, warnings, (400, f"Invalid numeric override for {field}")

    if len(weather_values) == len(ALL_WEATHER_FIELDS):
        return weather_values, "override", warnings, None

    dep_utc = req["scheduled_dep_utc"].to_pydatetime()
    arr_utc = dep_utc + timedelta(minutes=req["scheduled_duration_min"])
    now_utc = datetime.now(UTC)
    if dep_utc > now_utc + timedelta(hours=48):
        return (
            None,
            None,
            warnings,
            (422, "Departure is outside the 48-hour forecast window and weather_override is incomplete."),
        )

    latest = _load_latest_weather()
    if not latest:
        return None, None, warnings, (422, "Forecast weather unavailable and weather_override is incomplete.")

    airports_weather = latest.get("airports", latest)
    dep_hour_key = dep_utc.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    arr_hour_key = arr_utc.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    origin_hourly = airports_weather.get(req["origin_airport"], {}).get(dep_hour_key, {})
    dest_hourly = airports_weather.get(req["dest_airport"], {}).get(arr_hour_key, {})

    for base in WEATHER_BASE_FIELDS:
        o_key = f"origin_{base}"
        if o_key not in weather_values:
            if base not in origin_hourly or origin_hourly[base] is None:
                return None, None, warnings, (422, f"Missing forecast weather for {o_key} at {dep_hour_key}")
            weather_values[o_key] = float(origin_hourly[base])

        d_key = f"dest_{base}"
        if d_key not in weather_values:
            if base not in dest_hourly or dest_hourly[base] is None:
                return None, None, warnings, (422, f"Missing forecast weather for {d_key} at {arr_hour_key}")
            weather_values[d_key] = float(dest_hourly[base])

    source = "forecast+override" if override else "forecast"
    return weather_values, source, warnings, None


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    start = time.perf_counter()
    request_id = getattr(context, "aws_request_id", "local")

    try:
        artifacts = load_artifacts()
        payload, parse_err = _parse_payload(event)
        if parse_err:
            return _response(400, {"error": parse_err})

        req, val_err = _validate_and_parse_request(payload, artifacts["hubs"])
        if val_err:
            return _response(val_err[0], {"error": val_err[1]})

        weather, weather_source, warnings, weather_err = _resolve_weather(req)
        if weather_err:
            return _response(weather_err[0], {"error": weather_err[1]})

        metadata = artifacts["metadata"]
        if req["carrier"] not in metadata["priors"]["carrier"]:
            warnings.append(f"Unknown carrier '{req['carrier']}'; using global delay prior.")

        row = {
            "origin_airport": req["origin_airport"],
            "dest_airport": req["dest_airport"],
            "carrier": req["carrier"],
            "scheduled_dep_utc": req["scheduled_dep_utc"],
            "scheduled_duration_min": req["scheduled_duration_min"],
            "distance_miles": req["distance_miles"],
            **weather,
        }
        df = pd.DataFrame([row])
        df = add_temporal_features(df, artifacts["airports_csv"])
        df = apply_priors(df, metadata["priors"])
        X = df[FEATURE_COLUMNS]

        proba = float(artifacts["classifier"].predict_proba(X)[0, 1])
        threshold = float(metadata["threshold"])
        is_delayed = bool(proba >= threshold)

        if artifacts["severity_classifier"] is not None and artifacts["regressor_p50"] is not None:
            buckets = metadata.get("severity_buckets", ["15-45 min", "45-120 min", "120+ min"])
            sev_probs = artifacts["severity_classifier"].predict_proba(X)[0]
            severity_if_delayed = buckets[int(np.argmax(sev_probs))]
            predicted_bucket = severity_if_delayed if is_delayed else "on_time"
            bucket_probabilities = {"on_time": round(1.0 - proba, 4)}
            for label, sp in zip(buckets, sev_probs, strict=True):
                bucket_probabilities[label] = round(proba * float(sp), 4)

            delay_p50 = float(max(15.0, np.expm1(artifacts["regressor_p50"].predict(X)[0])))
            delay_p90 = float(max(delay_p50, np.expm1(artifacts["regressor_p90"].predict(X)[0])))
            est_delay = proba * delay_p50
        else:
            severity_if_delayed = None
            predicted_bucket = "delayed" if is_delayed else "on_time"
            bucket_probabilities = {"on_time": round(1.0 - proba, 4), "delayed": round(proba, 4)}
            est_delay = float(max(0.0, artifacts["regressor"].predict(X)[0]))
            delay_p50 = est_delay
            delay_p90 = est_delay

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        result = {
            "delay_probability": round(proba, 4),
            "is_delayed": is_delayed,
            "predicted_bucket": predicted_bucket,
            "severity_if_delayed": severity_if_delayed,
            "bucket_probabilities": bucket_probabilities,
            "delay_if_delayed_min": round(delay_p50, 1),
            "delay_if_delayed_p90_min": round(delay_p90, 1),
            "estimated_delay_min": round(est_delay, 2),
            "threshold": threshold,
            "model_version": metadata["model_version"],
            "weather_source": weather_source,
            "warnings": warnings,
            "latency_ms": latency_ms,
        }
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "origin": req["origin_airport"],
                    "dest": req["dest_airport"],
                    "carrier": req["carrier"],
                    "delay_probability": result["delay_probability"],
                    "is_delayed": is_delayed,
                    "model_version": metadata["model_version"],
                    "latency_ms": latency_ms,
                    "warnings": warnings,
                }
            )
        )
        return _response(200, result)
    except Exception:
        logger.exception("Unhandled error in PredictLambda (request_id=%s)", request_id)
        return _response(500, {"error": "Internal server error", "request_id": request_id})
