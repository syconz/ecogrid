import os
import requests
from datetime import datetime
from collections import defaultdict

GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/zip"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def get_weather_forecast(zip_code, country):
    """
    Real forecast via OpenWeatherMap:
      1) Geocode the zip+country to lat/lon (Geocoding API)
      2) Pull the 5-day / 3-hour forecast for that lat/lon
      3) Aggregate 3-hour blocks into daily day_temp (max) / night_temp (min)

    Returns a list of dicts shaped like the old mock data:
        [{"date": "Tue 19 Aug", "day_temp": 29.4, "night_temp": 21.1}, ...]
    Returns None if the API key is missing or any request fails, so the
    template's `{% if weather %}` check will just skip rendering it.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("[weather_api] OPENWEATHER_API_KEY not set in .env — skipping forecast.")
        return None

    if not zip_code or not country:
        return None

    try:
        # Step 1: geocode zip+country -> lat/lon
        geo_resp = requests.get(
            GEOCODE_URL,
            params={"zip": f"{zip_code},{country}", "appid": api_key},
            timeout=5,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        lat, lon = geo_data["lat"], geo_data["lon"]

        # Step 2: 5 day / 3 hour forecast for that location
        forecast_resp = requests.get(
            FORECAST_URL,
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
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

    # Step 3: group the 3-hour entries by calendar day, track min/max temp
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