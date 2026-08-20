import os
import requests
from datetime import datetime
from collections import defaultdict

GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/zip"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
SOLAR_URL = "https://api.open-meteo.com/v1/forecast"


def _geocode_zip(zip_code, country):
    """
    Resolves a zip code + ISO country code to (lat, lon) via OpenWeatherMap's
    Geocoding API. Shared by both the weather forecast and the solar
    irradiance fetch below, so we only write this logic once.
    Returns (lat, lon) or None on any failure.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key or not zip_code or not country:
        return None

    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"zip": f"{zip_code},{country}", "appid": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["lat"], data["lon"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"[weather_api] Geocoding failed: {e}")
        return None


def get_weather_forecast(zip_code, country):
    """
    Real forecast via OpenWeatherMap:
      1) Geocode the zip+country to lat/lon
      2) Pull the 5-day / 3-hour forecast for that lat/lon
      3) Aggregate 3-hour blocks into daily day_temp (max) / night_temp (min)

    Returns a list of dicts shaped like the old mock data:
        [{"date": "Tue 19 Aug", "day_temp": 29.4, "night_temp": 21.1}, ...]
    Returns None if geocoding or the forecast request fails, so the
    template's `{% if weather %}` check will just skip rendering it.
    """
    coords = _geocode_zip(zip_code, country)
    if not coords:
        return None
    lat, lon = coords

    try:
        forecast_resp = requests.get(
            FORECAST_URL,
            params={"lat": lat, "lon": lon, "appid": os.environ.get("OPENWEATHER_API_KEY"), "units": "metric"},
            timeout=5,
        )
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[weather_api] Request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"[weather_api] Unexpected response shape: {e}")
        return None

    days = defaultdict(list)
    for entry in forecast_data.get("list", []):
        dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
        days[dt.date()].append(entry["main"]["temp"])

    forecast = []
    for day, temps in sorted(days.items()):
        forecast.append({
            "date": day.strftime("%a %d %b"),
            "day_temp": round(max(temps), 1),
            "night_temp": round(min(temps), 1),
        })

    return forecast if forecast else None


def get_solar_irradiance_forecast(zip_code, country):
    """
    Real hourly solar irradiance (Global Horizontal Irradiance, W/m^2) for
    the next 24 hours via the Open-Meteo Solar Radiation API — free, no API
    key required. This is genuine measured/modeled solar data, unlike the
    old physically-approximated daylight bell curve.

    Returns a dict {hour_of_day (0-23): ghi_value_wm2}, or None if geocoding
    or the request fails, or if a full 24-hour set wasn't returned.
    """
    coords = _geocode_zip(zip_code, country)
    if not coords:
        return None
    lat, lon = coords

    try:
        resp = requests.get(
            SOLAR_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "shortwave_radiation",
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        times = data["hourly"]["time"]
        values = data["hourly"]["shortwave_radiation"]
    except requests.exceptions.RequestException as e:
        print(f"[weather_api] Solar irradiance request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"[weather_api] Unexpected solar response shape: {e}")
        return None

    curve = {}
    for t, v in zip(times, values):
        try:
            hour = int(t[11:13])  # ISO format "YYYY-MM-DDTHH:MM"
        except (ValueError, IndexError):
            continue
        curve[hour] = float(v) if v is not None else 0.0

    if len(curve) < 24:
        print(f"[weather_api] Incomplete solar curve ({len(curve)}/24 hours) — discarding.")
        return None

    return curve