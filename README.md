# EcoGrid ⚡

EcoGrid predicts building electricity usage and cost, forecasts a personalized
solar-vs-grid switching schedule, and helps you track your energy footprint
over time — built with Flask and trained on real building energy data.

A rebuild of an earlier Streamlit prototype ("Savergy"), with ideas drawn from
[SIH-SmartAutomation-Electricity-AI-Consumption](https://github.com/OmAmar106/SIH-SmartAutomation-Electricity-AI-Consumption).

## Features

- **Energy prediction** — a RandomForestRegressor trained on the real
  [ASHRAE Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction)
  dataset estimates hourly kWh usage and cost from building type + square footage.
- **Weather forecast** — real 5-day forecasts via the OpenWeatherMap API,
  geocoded from a zip code + ISO country code.
- **Renewable Switch Advisor** — 13 Prophet models (one per building type),
  trained on real historical hourly usage, forecast a 24-hour demand curve
  tied to your actual prediction. Combined with real hourly solar
  irradiance (Open-Meteo, geocoded from your zip/country) to recommend
  Solar vs. Grid hour by hour — falls back to an approximated daylight
  curve if location data isn't available.
- **SMS notifications** — sends your prediction summary via Twilio.
- **Accounts & history** — SQLite-backed signup/login (hashed passwords)
  with per-user prediction history driving the Analytics page.

## Honest caveats

- **Solar availability uses real irradiance data when available.** Hourly
  Global Horizontal Irradiance (GHI) is pulled live from the Open-Meteo
  Solar Radiation API, geocoded from the zip/country on your most recent
  prediction. If no location is on file, or the API call fails, it falls
  back to a physically-motivated daylight bell curve (peaking near solar
  noon) — the UI flags which one was used via the "Solar Data Source" badge
  on the Renewable Advisor page.
- **Twilio is on a trial account by default.** Trial accounts can only send
  one of Twilio's fixed template bodies (not custom text) and can only text
  numbers verified in the Twilio console. Upgrading the account removes both
  restrictions.
- **Potential Savings** (shown on the Predictions page) is a flat 12%
  heuristic, not a model output — the real model only predicts usage and cost.

## Setup

### Requirements
- Python 3.12 (newer versions may lack pre-built wheels for scikit-learn/numpy/pandas)
- A [Twilio](https://www.twilio.com/try-twilio) account (free trial is fine)
- An [OpenWeatherMap](https://openweathermap.org/api) API key (free tier)

### Install

```bash
git clone https://github.com/syconz/ecogrid.git
cd ecogrid

python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

This repo uses [Git LFS](https://git-lfs.github.com) to store the trained
models and datasets. If files under `models/` or `data/` look like small
text pointers instead of real data after cloning, install Git LFS and pull:

```bash
brew install git-lfs   # or your OS's equivalent
git lfs install
git lfs pull
```

### Configure environment variables

Create a `.env` file in the project root:

```
SECRET_KEY=generate_a_random_one_do_not_use_this_placeholder
OPENWEATHER_API_KEY=your_openweathermap_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
```

Generate a real `SECRET_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Run

```bash
python app.py
```

Visit `http://127.0.0.1:5000`, sign up for an account, and go from there.

## Regenerating the data/models (optional)

The trained model and dataset files are already included via Git LFS, so
this isn't required after a normal clone. Only needed if you want to retrain
from scratch (e.g. after changing `N_BUILDINGS_TO_SAMPLE` or feature columns):

```bash
python download_data.py          # requires a Kaggle account + API credentials
python clean_data.py
python train_model.py            # trains the RandomForest usage/cost model
python train_renewable_model.py  # trains the per-building-type Prophet models
```

## Project structure

```
ecogrid/
├── app.py                    # Flask routes
├── db.py                     # SQLite: users + prediction history
├── config.py
├── download_data.py           # pulls ASHRAE dataset via Kaggle API
├── clean_data.py               # cleans/prepares the dataset
├── train_model.py              # trains the RandomForest usage/cost model
├── train_renewable_model.py    # trains per-building-type Prophet models
├── templates/                  # Jinja2 templates
├── static/                     # CSS/JS
├── utils/
│   ├── ml_model.py             # loads models, predict_energy_cost(), get_renewable_advice()
│   ├── weather_api.py          # OpenWeatherMap integration
│   └── twilio_api.py           # Twilio SMS integration
├── data/                       # cleaned datasets (Git LFS)
└── models/                     # trained .pkl models (Git LFS)
```

## Not yet built

- Hardware integration (ESP32 + ACS712 current sensor via MQTT/HTTP) for
  live sensor readings instead of predictions — planned but not started.

gunicorn