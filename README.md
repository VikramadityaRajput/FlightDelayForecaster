# FlightDelayForecaster
Predicts whether a US domestic flight between major hubs will arrive 15+ minutes late, using schedule, route, historical, and weather data. Trained locally (scikit-learn + LightGBM) and served serverless on AWS (Lambda + API Gateway + S3 + EventBridge via SAM).

See [FlightDelay.md](FlightDelay.md) for the full spec and daily milestones.

## Setup
```bash
brew install libomp   # required by LightGBM on macOS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
