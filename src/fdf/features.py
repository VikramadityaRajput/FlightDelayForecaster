from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["origin_airport", "dest_airport", "carrier"]

NUMERIC_FEATURES = [
    "dep_hour_sin",
    "dep_hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month",
    "scheduled_duration_min",
    "distance_miles",
    "origin_hour_delay_rate_prior",
    "carrier_delay_rate_prior",
    "route_delay_rate_prior",
    "origin_temp_c",
    "origin_wind_speed_kmh",
    "origin_wind_gusts_kmh",
    "origin_precipitation_mm",
    "origin_snowfall_cm",
    "origin_cloud_cover_pct",
    "dest_temp_c",
    "dest_wind_speed_kmh",
    "dest_wind_gusts_kmh",
    "dest_precipitation_mm",
    "dest_snowfall_cm",
    "dest_cloud_cover_pct",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LEAKAGE_BLOCKLIST = {
    "DepDelay",
    "DepDelayMinutes",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrTime",
    "DepTime",
    "WheelsOff",
    "WheelsOn",
    "TaxiOut",
    "TaxiIn",
    "ActualElapsedTime",
    "AirTime",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "target_delay_min",
    "is_delayed",
}

BTS_RENAME_MAP = {
    "Origin": "origin_airport",
    "Dest": "dest_airport",
    "Reporting_Airline": "carrier",
    "CRSElapsedTime": "scheduled_duration_min",
    "Distance": "distance_miles",
    "ArrDelayMinutes": "target_delay_min",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw BTS column names to the shared API/feature schema."""
    return df.rename(columns={k: v for k, v in BTS_RENAME_MAP.items() if k in df.columns}).copy()


def add_temporal_features(df: pd.DataFrame, airports_csv: Path) -> pd.DataFrame:
    """Add cyclical local-time features from scheduled_dep_utc and origin_airport."""
    out = standardize_columns(df)
    airports = pd.read_csv(airports_csv)
    tz_map = dict(zip(airports["code"], airports["tz"]))

    dep_utc = pd.to_datetime(out["scheduled_dep_utc"], utc=True)
    origin_tz = out["origin_airport"].map(tz_map)
    if origin_tz.isna().any():
        unknown = sorted(out.loc[origin_tz.isna(), "origin_airport"].unique())
        raise ValueError(f"Unknown origin airport(s): {unknown}")

    # tz_convert works on one timezone at a time, so convert each tz group separately.
    local_hour = pd.Series(0, index=out.index, dtype="int64")
    local_dow = pd.Series(0, index=out.index, dtype="int64")
    local_month = pd.Series(0, index=out.index, dtype="int64")
    for tz in origin_tz.unique():
        mask = origin_tz == tz
        local = dep_utc[mask].dt.tz_convert(ZoneInfo(tz))
        local_hour[mask] = local.dt.hour
        local_dow[mask] = local.dt.day_of_week  # Monday=0 ... Sunday=6
        local_month[mask] = local.dt.month

    out["dep_hour"] = local_hour  # raw hour kept for the origin_hour prior lookup
    out["dep_hour_sin"] = np.sin(2 * np.pi * local_hour / 24.0)
    out["dep_hour_cos"] = np.cos(2 * np.pi * local_hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * local_dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * local_dow / 7.0)
    out["is_weekend"] = (local_dow >= 5).astype(int)
    out["month"] = local_month
    return out


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by scheduled_dep_utc and split into train / val / test without shuffling.

    Cuts happen at timestamp boundaries, so flights sharing a departure time never
    straddle two splits (train max < val min < test min).
    """
    if df["scheduled_dep_utc"].isna().any():
        raise ValueError("scheduled_dep_utc has missing values; drop them before splitting.")

    ordered = df.sort_values("scheduled_dep_utc", kind="stable").reset_index(drop=True)
    t = ordered["scheduled_dep_utc"]
    n_train = int(len(ordered) * train_frac)
    n_val = int(len(ordered) * val_frac)
    cut_val = t.iloc[n_train]
    cut_test = t.iloc[n_train + n_val]

    train = ordered[t < cut_val].copy()
    val = ordered[(t >= cut_val) & (t < cut_test)].copy()
    test = ordered[t >= cut_test].copy()
    return train, val, test


def _make_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Join one or more columns into a string key, e.g. 'ORD|18' or 'ORD|EWR'."""
    key = df[cols[0]].astype(str)
    for col in cols[1:]:
        key = key + "|" + df[col].astype(str)
    return key


def _smoothed_rate(
    df: pd.DataFrame, group_cols: list[str], global_mean: float, m: int, target_col: str = "is_delayed"
) -> dict[str, float]:
    """(delayed_count + m * global_mean) / (count + m) for each group, as {key: rate}."""
    stats = df.groupby(_make_key(df, group_cols))[target_col].agg(["sum", "count"])
    rates = (stats["sum"] + m * global_mean) / (stats["count"] + m)
    return {str(k): float(v) for k, v in rates.items()}


PRIOR_SPECS = {
    "origin_hour_delay_rate_prior": ("origin_hour", ["origin_airport", "dep_hour"]),
    "carrier_delay_rate_prior": ("carrier", ["carrier"]),
    "route_delay_rate_prior": ("route", ["origin_airport", "dest_airport"]),
}


def fit_priors(train_df: pd.DataFrame, m: int = 20) -> dict:
    """Compute smoothed historical delay rates STRICTLY from train_df.

    Returns a JSON-serializable dict so it can be saved in feature_metadata.json
    and loaded by the inference Lambda.
    """
    global_mean = float(train_df["is_delayed"].mean())
    priors = {"global_mean": global_mean, "m": m}
    for name, cols in PRIOR_SPECS.values():
        priors[name] = _smoothed_rate(train_df, cols, global_mean, m)
    return priors


def apply_priors(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Map fitted historical priors onto df, falling back to global_mean for unseen keys."""
    out = df.copy()
    default = priors["global_mean"]
    for feature, (name, cols) in PRIOR_SPECS.items():
        out[feature] = _make_key(out, cols).map(priors[name]).fillna(default).astype(float)
    return out
