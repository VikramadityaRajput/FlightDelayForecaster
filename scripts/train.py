"""
Usage:
    python scripts/train.py              # train, evaluate, save models/<version>/
    python scripts/train.py --ablation   # also compare feature sets (priors / month on vs off)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: F401  
from sklearn.compose import ColumnTransformer  # noqa: F401
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: F401
from sklearn.metrics import (  # noqa: F401
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: F401

from fdf.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_temporal_features,
    apply_priors,
    chronological_split,
    fit_priors,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interim" / "flights_with_weather.parquet"
AIRPORTS_CSV = ROOT / "config" / "airports.csv"
MODELS_DIR = ROOT / "models"

CLS_TARGET = "is_delayed"
REG_TARGET = "target_delay_min"
PRIOR_FEATURES = [
    "origin_hour_delay_rate_prior",
    "carrier_delay_rate_prior",
    "route_delay_rate_prior",
]



def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_parquet(DATA_PATH).dropna(subset=["scheduled_dep_utc"])
    df = add_temporal_features(df, AIRPORTS_CSV)
    train, val, test = chronological_split(df)
    priors = fit_priors(train)  # train only!
    train, val, test = (apply_priors(x, priors) for x in (train, val, test))
    return train, val, test, priors


# =============================================================== TODO 1-2: preprocessing


def make_linear_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Preprocessing for linear models (LogisticRegression, Ridge)."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )


def make_tree_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Preprocessing for LightGBM."""
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ]
    )


def build_classifiers(categorical: list[str], numeric: list[str], pos_weight: float) -> dict[str, Pipeline]:
    """Return {"logreg": Pipeline, "lgbm": Pipeline}."""
    return {
        "logreg": Pipeline(
            [
                ("prep", make_linear_preprocessor(categorical, numeric)),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        ),
        "lgbm": Pipeline(
            [
                ("prep", make_tree_preprocessor(categorical, numeric)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        scale_pos_weight=pos_weight,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def build_regressors(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    """Return {"ridge": Pipeline, "lgbm": Pipeline} predicting delay minutes."""
    return {
        "ridge": Pipeline(
            [
                ("prep", make_linear_preprocessor(categorical, numeric)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "lgbm": Pipeline(
            [
                ("prep", make_tree_preprocessor(categorical, numeric)),
                (
                    "model",
                    LGBMRegressor(
                        objective="l1",
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def best_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Pick the probability cutoff that maximizes F1. Call this on VALIDATION data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    """Compute binary classification and calibration metrics."""
    pred = proba >= threshold
    return {
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "base_rate": round(float(y_true.mean()), 4),
    }


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict:
    """Compute delay-minute regression error metrics."""
    return {
        "mae": round(float(mean_absolute_error(y_true, pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, pred)), 2),
        "median_ae": round(float(np.median(np.abs(y_true - pred))), 2),
    }


# =============================================================== training loop (done)


def run(train, val, test, categorical, numeric, verbose=True) -> dict:
    """Fit all models on train, tune threshold on val, report val + test metrics."""
    results = {"classifiers": {}, "regressors": {}}
    fitted = {}

    pos_weight = float((train[CLS_TARGET] == 0).sum() / (train[CLS_TARGET] == 1).sum())
    features = categorical + numeric

    for name, pipe in build_classifiers(categorical, numeric, pos_weight).items():
        pipe.fit(train[features], train[CLS_TARGET])
        val_proba = pipe.predict_proba(val[features])[:, 1]
        test_proba = pipe.predict_proba(test[features])[:, 1]
        threshold = best_f1_threshold(val[CLS_TARGET], val_proba)
        results["classifiers"][name] = {
            "threshold": round(threshold, 4),
            "val": classification_metrics(val[CLS_TARGET], val_proba, threshold),
            "test": classification_metrics(test[CLS_TARGET], test_proba, threshold),
        }
        fitted[f"cls_{name}"] = pipe

    # Naive regression baseline: always predict the training median delay.
    median_pred = float(train[REG_TARGET].median())
    results["regressors"]["predict_median"] = {
        "val": regression_metrics(val[REG_TARGET], np.full(len(val), median_pred)),
        "test": regression_metrics(test[REG_TARGET], np.full(len(test), median_pred)),
    }
    for name, pipe in build_regressors(categorical, numeric).items():
        pipe.fit(train[features], train[REG_TARGET])
        results["regressors"][name] = {
            "val": regression_metrics(val[REG_TARGET], np.clip(pipe.predict(val[features]), 0, None)),
            "test": regression_metrics(test[REG_TARGET], np.clip(pipe.predict(test[features]), 0, None)),
        }
        fitted[f"reg_{name}"] = pipe

    if verbose:
        print_results(results)
    return {"results": results, "fitted": fitted}


def print_results(results: dict) -> None:
    rows = []
    for name, r in results["classifiers"].items():
        for split in ("val", "test"):
            rows.append({"model": name, "split": split, **r[split], "threshold": r["threshold"]})
    print("\n=== Classification (is_delayed) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    rows = [{"model": n, "split": s, **r[s]} for n, r in results["regressors"].items() for s in ("val", "test")]
    print("\n=== Regression (target_delay_min) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def run_ablation(train, val, test) -> pd.DataFrame:
    """Tradeoffs #1 and #2: how much do the priors and `month` actually help?"""
    variants = {
        "all features": NUMERIC_FEATURES,
        "no priors": [f for f in NUMERIC_FEATURES if f not in PRIOR_FEATURES],
        "no month": [f for f in NUMERIC_FEATURES if f != "month"],
    }
    rows = []
    for label, numeric in variants.items():
        out = run(train, val, test, CATEGORICAL_FEATURES, numeric, verbose=False)
        lgbm = out["results"]["classifiers"]["lgbm"]
        # Train PR-AUC vs val PR-AUC: a big gap = overfitting / in-sample prior leakage.
        train_proba = out["fitted"]["cls_lgbm"].predict_proba(train[CATEGORICAL_FEATURES + numeric])[:, 1]
        rows.append(
            {
                "variant": label,
                "train_pr_auc": round(average_precision_score(train[CLS_TARGET], train_proba), 4),
                "val_pr_auc": lgbm["val"]["pr_auc"],
                "test_pr_auc": lgbm["test"]["pr_auc"],
            }
        )
    table = pd.DataFrame(rows)
    print("\n=== Ablation (LightGBM classifier) ===")
    print(table.to_string(index=False))
    return table


# =============================================================== save artifacts (done)


def save_artifacts(out: dict, train: pd.DataFrame, priors: dict, numeric: list[str]) -> Path:
    version = f"{date.today().isoformat()}-lgbm-v1"
    out_dir = MODELS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    results = out["results"]
    threshold = results["classifiers"]["lgbm"]["threshold"]
    joblib.dump(
        {"classifier": out["fitted"]["cls_lgbm"], "regressor": out["fitted"]["reg_lgbm"]},
        out_dir / "model.joblib",
    )
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    metadata = {
        "model_version": version,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": numeric,
        "threshold": threshold,
        "priors": priors,
        "hubs": sorted(pd.read_csv(AIRPORTS_CSV)["code"]),
        "train_date_range": [str(train["scheduled_dep_utc"].min()), str(train["scheduled_dep_utc"].max())],
        "library_versions": {
            "python_sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
        },
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved artifacts to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation", action="store_true", help="compare feature sets")
    args = parser.parse_args()

    train, val, test, priors = load_splits()
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

    out = run(train, val, test, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    save_artifacts(out, train, priors, NUMERIC_FEATURES)

    if args.ablation:
        run_ablation(train, val, test)


if __name__ == "__main__":
    main()

