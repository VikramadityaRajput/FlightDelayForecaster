# FlightDelayForecaster

Predicts arrival delay risk (&ge; 15 min), delay severity buckets (`On Time`, `15-45 min`, `45-120 min`, `120+ min`), and conditional delay ranges (`p50` / `p90`) for US domestic flights between 15 major hubs using schedule, historical priors, and live Open-Meteo weather forecasts.

- **Live Demo:** https://vikramadityarajput.github.io/FlightDelayForecaster/
- **Design & Tradeoffs:** [Tradeoffs.md](Tradeoffs.md) · [FlightDelay.md](FlightDelay.md)

## Architecture
- **Training (local):** Python 3.12, scikit-learn, LightGBM on ~384k BTS On-Time Performance flights (Nov 2024 – Mar 2025) joined with Open-Meteo historical hourly weather. Chronological 70/15/15 split with train-only smoothed target priors and leakage tests.
- **Two-Stage Hurdle Model (`lgbm-v2`):**
  - **Stage 1:** Binary LightGBM classifier for `P(delay >= 15 min)` across all flights.
  - **Stage 2:** Trained **only on delayed flights** — multiclass LightGBM for severity buckets (`15-45`, `45-120`, `120+ min`) + conditional quantile regressors (`p50`, `p90` on `log1p` delay).
- **Serverless AWS Stack (SAM):**
  - `IngestFunction` (Python 3.12 Lambda + EventBridge 15-min schedule) fetches 48h Open-Meteo forecasts for all 15 hubs into S3 (`features/latest_weather.json` + 30-day expiring `raw/` snapshots).
  - `PredictFunction` (ARM64 container Lambda + HTTP API Gateway) serves `POST /predict` (~200–250 ms warm latency, throttled at 5 req/s).
  - Static frontend hosted on GitHub Pages (`docs/index.html`).

## Held-Out Test Metrics (Chronological Split)
| Stage | Task | Model | Test Metric | Baseline |
|---|---|---|---|---|
| Stage 1 | Delay &ge; 15 min | LightGBM | PR-AUC **0.3848**, F1 **0.4270** | Base rate 0.2238 |
| Stage 2a | Severity buckets (delayed only) | LightGBM | Macro-F1 **0.2907**, Log-loss **0.9915** | Prior baseline 0.2273 / 1.0001 |
| Stage 2b | Conditional `p50` (delayed only) | LightGBM Quantile | MAE **46.17**, MedAE **19.52** | Delayed median 47.06 / 20.00 |

## Local Setup & Tests
```bash
brew install libomp   # required by LightGBM on macOS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest -v             # 23 leakage + offline/handler parity tests
python scripts/train.py
```

## Deploy (AWS SAM)
```bash
sam build && sam deploy --guided
```
