from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fdf.features import (
    FEATURE_COLUMNS,
    LEAKAGE_BLOCKLIST,
    add_temporal_features,
    apply_priors,
    chronological_split,
    fit_priors,
)

ROOT = Path(__file__).resolve().parents[1]
AIRPORTS_CSV = ROOT / "config" / "airports.csv"
REAL_DATA = ROOT / "data" / "interim" / "flights_with_weather.parquet"


def make_flights(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic flights: one every 30 min, with some duplicate timestamps."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC")
    times = times.repeat(2)[:n]  # pairs of flights share a departure time
    return pd.DataFrame(
        {
            "scheduled_dep_utc": times,
            "origin_airport": rng.choice(["ORD", "ATL", "DEN"], n),
            "dest_airport": rng.choice(["JFK", "LAX", "SFO"], n),
            "carrier": rng.choice(["UA", "DL", "AA"], n),
            "is_delayed": rng.integers(0, 2, n),
        }
    )


# split


def test_split_is_strictly_chronological():
    train, val, test = chronological_split(make_flights())
    assert train["scheduled_dep_utc"].max() < val["scheduled_dep_utc"].min()
    assert val["scheduled_dep_utc"].max() < test["scheduled_dep_utc"].min()


def test_split_keeps_every_row_exactly_once():
    df = make_flights()
    train, val, test = chronological_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_rough_proportions():
    train, val, _ = chronological_split(make_flights(n=1000))
    n = 1000
    assert abs(len(train) / n - 0.70) < 0.02
    assert abs(len(val) / n - 0.15) < 0.02


def test_split_rejects_missing_timestamps():
    df = make_flights()
    df.loc[5, "scheduled_dep_utc"] = pd.NaT
    with pytest.raises(ValueError):
        chronological_split(df)


#leakage columns


def test_no_blocklisted_column_is_a_feature():
    assert not set(FEATURE_COLUMNS) & LEAKAGE_BLOCKLIST


def test_targets_are_not_features():
    assert "is_delayed" not in FEATURE_COLUMNS
    assert "target_delay_min" not in FEATURE_COLUMNS


#priors


def _with_temporal(df: pd.DataFrame) -> pd.DataFrame:
    return add_temporal_features(df, AIRPORTS_CSV)


def test_priors_ignore_val_and_test_labels():
    """Flipping every label outside train must not change the fitted priors."""
    train, val, test = chronological_split(_with_temporal(make_flights()))
    priors_before = fit_priors(train)

    val["is_delayed"] = 1 - val["is_delayed"]
    test["is_delayed"] = 1 - test["is_delayed"]
    priors_after = fit_priors(train)

    assert priors_before == priors_after


def test_carrier_prior_matches_manual_smoothing():
    train, _, _ = chronological_split(_with_temporal(make_flights()))
    m = 20
    priors = fit_priors(train, m=m)

    g = train["is_delayed"].mean()
    ua = train[train["carrier"] == "UA"]["is_delayed"]
    expected = (ua.sum() + m * g) / (len(ua) + m)
    assert priors["carrier"]["UA"] == pytest.approx(expected)


def test_unseen_keys_fall_back_to_global_mean():
    train, _, _ = chronological_split(_with_temporal(make_flights()))
    priors = fit_priors(train)

    unseen = _with_temporal(
        pd.DataFrame(
            {
                "scheduled_dep_utc": ["2025-06-01T12:00:00Z"],
                "origin_airport": ["MIA"],  # never appears in synthetic train
                "dest_airport": ["SEA"],
                "carrier": ["ZZ"],
            }
        )
    )
    out = apply_priors(unseen, priors)
    for col in ["origin_hour_delay_rate_prior", "carrier_delay_rate_prior", "route_delay_rate_prior"]:
        assert out[col].iloc[0] == pytest.approx(priors["global_mean"])


# temporal features


def _one_flight(origin: str, dep_utc: str) -> pd.Series:
    df = pd.DataFrame(
        {"scheduled_dep_utc": [dep_utc], "origin_airport": [origin], "dest_airport": ["JFK"], "carrier": ["UA"]}
    )
    return add_temporal_features(df, AIRPORTS_CSV).iloc[0]


def test_local_hour_uses_origin_timezone_winter():
    # Chicago is UTC-6 in November: 18:30 UTC -> 12:30 local
    assert _one_flight("ORD", "2026-11-25T18:30:00Z")["dep_hour"] == 12


def test_local_hour_respects_daylight_saving():
    # Chicago is UTC-5 in July: 18:30 UTC -> 13:30 local
    assert _one_flight("ORD", "2026-07-15T18:30:00Z")["dep_hour"] == 13


def test_weekend_flag_uses_local_date():
    # 05:00 UTC Monday is still Sunday 21:00 in Los Angeles -> weekend locally.
    assert _one_flight("LAX", "2025-01-13T05:00:00Z")["is_weekend"] == 1
    # 05:00 UTC Saturday is still Friday 21:00 in Los Angeles -> NOT weekend locally.
    assert _one_flight("LAX", "2025-01-11T05:00:00Z")["is_weekend"] == 0


def test_cyclical_hour_wraps_around_midnight():
    late = _one_flight("ORD", "2025-01-15T05:00:00Z")  # 23:00 local
    early = _one_flight("ORD", "2025-01-15T06:00:00Z")  # 00:00 local
    noon = _one_flight("ORD", "2025-01-15T18:00:00Z")  # 12:00 local

    def dist(a, b):
        return np.hypot(a["dep_hour_sin"] - b["dep_hour_sin"], a["dep_hour_cos"] - b["dep_hour_cos"])

    assert dist(late, early) < dist(late, noon)


def test_unknown_airport_raises():
    with pytest.raises(ValueError):
        _one_flight("XXX", "2025-01-15T12:00:00Z")


# -real data (optional)


@pytest.mark.skipif(not REAL_DATA.exists(), reason="local dataset not built")
def test_real_dataset_end_to_end():
    df = pd.read_parquet(REAL_DATA).dropna(subset=["scheduled_dep_utc"])
    df = add_temporal_features(df, AIRPORTS_CSV)
    train, val, test = chronological_split(df)

    assert train["scheduled_dep_utc"].max() < val["scheduled_dep_utc"].min()
    assert val["scheduled_dep_utc"].max() < test["scheduled_dep_utc"].min()

    priors = fit_priors(train)
    for split in (train, val, test):
        feats = apply_priors(split, priors)[FEATURE_COLUMNS]
        assert feats.isna().sum().sum() == 0
