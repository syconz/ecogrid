"""
utils/ml_model.py
Loads the trained RandomForest model (models/energy_model.pkl) and exposes
predict_energy_cost() for the Flask app to call.

get_historical_usage() is UNCHANGED from before — still mock/random data,
to be replaced in later steps.

MEMORY FIX: get_renewable_advice() used to load and permanently cache all
13 Prophet models at once, which was pushing memory usage past Render's
512MB free-tier limit and crashing the service. It now keeps at most ONE
Prophet model in memory at a time (see _load_renewable_model_for below).
"""

import os
import pickle
import random
import pandas as pd
import math
import datetime
from utils.weather_api import get_solar_irradiance_forecast

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "energy_model.pkl")

RENEWABLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "renewable_prophet_models.pkl")


def _solar_availability_curve():
    """
    Physically-motivated approximation of solar output across 24 hours.
    NOT derived from real solar irradiance data (the training dataset has
    none) — this is a daylight bell curve: zero before sunrise/after sunset,
    peaking near solar noon (hour 13). Scaled 0-100 to match the demand
    curve's percentage-of-peak scale for a fair hour-by-hour comparison.
    """
    curve = {}
    for h in range(24):
        if 6 <= h <= 19:  # approximate daylight window
            # bell shape peaking at hour 13
            curve[h] = max(0, 100 * math.cos((h - 13) * math.pi / 14))
        else:
            curve[h] = 0.0
    return curve

# Cost assumption — adjust to match your local electricity rate.
# This is a placeholder; a real app might make this configurable per region.
COST_PER_KWH = 8.5  # e.g. INR per kWh — change as needed

_model_artifact = None  # loaded once, cached at module level (RandomForest — small, fine to keep)


def _load_model():
    """Loads the pickled RandomForest model artifact once and caches it in memory."""
    global _model_artifact
    if _model_artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run train_model.py first to generate it."
            )
        with open(MODEL_PATH, "rb") as f:
            _model_artifact = pickle.load(f)
    return _model_artifact


def get_available_building_types():
    """
    Returns the list of building types the model was actually trained on.
    Use this to populate the dropdown in predictions.html so users can only
    pick categories the model understands.
    """
    artifact = _load_model()
    return artifact["category_values"]


def predict_energy_cost(
    square_feet,
    building_type,
    air_temperature=None,
    dew_temperature=None,
    precip_depth_1_hr=None,
    sea_level_pressure=None,
    wind_direction=None,
    wind_speed=None,
    hour=None,
    is_weekend=None,
):
    """
    Predicts hourly electricity usage (kWh) and estimated cost for a building.

    Required:
        square_feet   (float) — building size in square feet
        building_type (str)   — must match one of get_available_building_types()

    Optional (weather/time context — real weather API not wired up yet,
    so sensible defaults are used if omitted):
        air_temperature, dew_temperature, precip_depth_1_hr,
        sea_level_pressure, wind_direction, wind_speed (all floats)
        hour (int, 0-23) — defaults to current hour if not given
        is_weekend (0 or 1) — defaults to checking today's actual weekday

    Returns:
        dict with:
            predicted_usage_kwh (float)
            estimated_cost (float)
            building_type_used (str) — what was actually matched/used
    """
    artifact = _load_model()
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    known_categories = artifact["category_values"]

    # Fill in sensible defaults for anything not provided (placeholder until
    # real weather API integration replaces these)
    if air_temperature is None:
        air_temperature = 20.0
    if dew_temperature is None:
        dew_temperature = 10.0
    if precip_depth_1_hr is None:
        precip_depth_1_hr = 0.0
    if sea_level_pressure is None:
        sea_level_pressure = 1016.0
    if wind_direction is None:
        wind_direction = 180.0
    if wind_speed is None:
        wind_speed = 3.0
    if hour is None:
        hour = datetime.datetime.now().hour
    if is_weekend is None:
        is_weekend = 1 if datetime.datetime.now().weekday() >= 5 else 0

    # Validate building_type against what the model actually knows
    if building_type not in known_categories:
        # Fall back to "Office" (or the first known category) rather than
        # crashing, but flag it so it's visible during testing.
        fallback = "Office" if "Office" in known_categories else known_categories[0]
        print(
            f"WARNING: building_type '{building_type}' not recognized by model. "
            f"Falling back to '{fallback}'. Known types: {known_categories}"
        )
        building_type = fallback

    # Build a single-row DataFrame matching the exact feature columns the
    # model was trained on (numeric features + one-hot building type columns)
    row = {col: 0 for col in feature_cols}
    row["square_feet"] = square_feet
    row["air_temperature"] = air_temperature
    row["dew_temperature"] = dew_temperature
    row["precip_depth_1_hr"] = precip_depth_1_hr
    row["sea_level_pressure"] = sea_level_pressure
    row["wind_direction"] = wind_direction
    row["wind_speed"] = wind_speed
    row["hour"] = hour
    row["is_weekend"] = is_weekend

    type_col = f"use_{building_type}"
    if type_col in row:
        row[type_col] = 1

    X = pd.DataFrame([row], columns=feature_cols)

    predicted_usage_kwh = float(model.predict(X)[0])
    estimated_cost = round(predicted_usage_kwh * COST_PER_KWH, 2)

    return {
        "predicted_usage_kwh": round(predicted_usage_kwh, 2),
        "estimated_cost": estimated_cost,
        "building_type_used": building_type,
    }


# ---------------------------------------------------------------------------
# Renewable Switch Advisor — Prophet models
#
# MEMORY FIX: the pickle at RENEWABLE_MODEL_PATH bundles all 13 per-building
# -type Prophet models into one file. The old code unpickled it once and
# cached the whole dict of 13 models in memory forever — that's what was
# pushing the app past Render's free-tier 512MB limit and crashing it.
#
# Now we cache at most ONE Prophet model at a time. Unpickling still briefly
# loads all 13 into memory (that's unavoidable given how the file is
# structured), but everything except the requested model is dropped right
# after, so steady-state memory holds just 1 model instead of 13. Switching
# building types re-reads the pickle — a small CPU cost — in exchange for a
# much smaller long-term memory footprint.
# ---------------------------------------------------------------------------

_renewable_building_types = None  # list of known types, cached (cheap — just strings)
_cached_model_type = None         # which single building type is currently cached
_cached_model = None              # the one Prophet model currently held in memory


def _get_renewable_building_types():
    """
    Returns the list of building types the renewable models were trained on,
    without permanently caching the (heavy) Prophet models themselves.
    """
    global _renewable_building_types
    if _renewable_building_types is None:
        if not os.path.exists(RENEWABLE_MODEL_PATH):
            raise FileNotFoundError(
                f"Renewable forecasting models not found at {RENEWABLE_MODEL_PATH}. "
                "Run train_renewable_model.py first to generate them."
            )
        with open(RENEWABLE_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        _renewable_building_types = artifact["building_types"]
        del artifact  # only needed the type list; let the other models get GC'd
    return _renewable_building_types


def _load_renewable_model_for(building_type):
    """Loads (and caches) only the Prophet model for the requested building_type."""
    global _cached_model_type, _cached_model

    if _cached_model_type == building_type and _cached_model is not None:
        return _cached_model

    if not os.path.exists(RENEWABLE_MODEL_PATH):
        raise FileNotFoundError(
            f"Renewable forecasting models not found at {RENEWABLE_MODEL_PATH}. "
            "Run train_renewable_model.py first to generate them."
        )

    with open(RENEWABLE_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    models = artifact["models"]
    if building_type not in models:
        del models, artifact
        raise KeyError(f"No renewable model for building_type '{building_type}'")

    _cached_model = models[building_type]
    _cached_model_type = building_type

    del models
    del artifact  # drop references to the other 12 models so GC can reclaim them

    return _cached_model


def get_renewable_advice(usage_kwh=450, building_type=None, zip_code=None, country=None):
    """
    Builds a 24-hour Solar-vs-Grid recommendation table.

    Demand curve: from a REAL Prophet model trained on actual historical
    hourly usage for the given building_type, scaled to the user's actual
    predicted daily usage (see train_renewable_model.py).

    Solar curve: REAL hourly solar irradiance (GHI) via Open-Meteo, when a
    zip_code + country are provided and the API call succeeds. Falls back
    to a physically-motivated daylight bell curve (documented in
    _solar_availability_curve()) if no location is given or the real fetch
    fails — this fallback is flagged via the returned "solar_is_real" key
    so the UI can be honest about which one was used.

    Both curves are normalized to a 0-100 percent-of-peak scale so they're
    directly comparable hour-by-hour, matching what the template displays.
    """
    available_types = _get_renewable_building_types()

    if building_type not in available_types:
        fallback = "Office" if "Office" in available_types else available_types[0]
        print(
            f"WARNING: No renewable model for building_type '{building_type}'. "
            f"Falling back to '{fallback}'. Available: {available_types}"
        )
        building_type = fallback

    model = _load_renewable_model_for(building_type)

    future = model.make_future_dataframe(periods=24, freq="h")
    forecast = model.predict(future)
    next_24 = forecast.tail(24)["yhat"].clip(lower=0).reset_index(drop=True)

    total_shape = next_24.sum()
    if total_shape <= 0:
        hourly_kwh = [usage_kwh / 24] * 24
    else:
        hourly_kwh = [(v / total_shape) * usage_kwh for v in next_24]

    peak_demand = max(hourly_kwh) if max(hourly_kwh) > 0 else 1
    demand_pct = [round(float(v / peak_demand) * 100, 1) for v in hourly_kwh]

    # Try real solar irradiance first; fall back to the approximated curve
    solar_is_real = False
    real_irradiance = None
    if zip_code and country:
        real_irradiance = get_solar_irradiance_forecast(zip_code, country)

    if real_irradiance:
        peak_irradiance = max(real_irradiance.values()) or 1
        solar_curve = {h: round((v / peak_irradiance) * 100, 1) for h, v in real_irradiance.items()}
        solar_is_real = True
    else:
        solar_curve = _solar_availability_curve()

    hours = []
    solar_hour_count = 0

    for h in range(24):
        solar_pct = round(solar_curve.get(h, 0.0), 1)
        demand = demand_pct[h]
        recommendation = "Solar" if solar_pct >= demand else "Grid"
        if recommendation == "Solar":
            solar_hour_count += 1

        hours.append({
            "hour": f"{h:02d}:00",
            "solar_available": solar_pct,
            "demand": demand,
            "recommendation": recommendation,
        })

    best_hour = max(hours, key=lambda x: x["solar_available"])
    best_hour_num = int(best_hour["hour"].split(":")[0])

    return {
        "best_window": f"{best_hour_num:02d}:00 - {(best_hour_num + 1) % 24:02d}:00",
        "solar_hour_count": solar_hour_count,
        "hours": hours,
        "solar_is_real": solar_is_real,
    }


def get_historical_usage(days=30):
    """MOCK — simulates the last `days` days of usage history for analytics.
    Still random data, but each entry now carries a real calendar date
    (most recent day last) so the Analytics chart can label its axis
    properly instead of showing generic day numbers."""
    data = []
    today = datetime.date.today()

    for i in range(days - 1, -1, -1):  # oldest first, today last
        day_date = today - datetime.timedelta(days=i)
        data.append({
            "date": day_date.strftime("%b %d"),  # e.g. "Aug 14"
            "usage_kwh": round(random.uniform(15, 45), 1),
        })

    avg_usage = round(sum(d["usage_kwh"] for d in data) / len(data), 1)
    total_cost = round(sum(d["usage_kwh"] for d in data) * COST_PER_KWH, 2)
    return {
        "daily_data": data,
        "avg_usage": avg_usage,
        "total_cost": total_cost,
    }