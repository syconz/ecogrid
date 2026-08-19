"""
train_renewable_model.py
Trains one Prophet model per building type on the cleaned ASHRAE dataset,
capturing the real daily usage pattern for that category of building.

Unlike train_model.py (which predicts a single total usage number from
static inputs), this captures SHAPE over time — i.e. what fraction of a
day's usage happens at each hour — which is what the Renewable Advisor
needs to build an hour-by-hour demand curve.

Run once (or whenever training data changes):
    python train_renewable_model.py

Output:
    models/renewable_prophet_models.pkl
        {
          "models": {building_type: Prophet model, ...},
          "building_types": [list of types that got a model],
        }
"""

import os
import pickle
import pandas as pd
from prophet import Prophet

INPUT_PATH = "data/training_ready.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "renewable_prophet_models.pkl")

# Building types with too few rows produce unreliable Prophet fits.
MIN_ROWS_PER_TYPE = 500


def build_hourly_series_per_type(df):
    """
    For each building type, averages meter_reading across all buildings of
    that type at each timestamp, producing one clean hourly series per type.
    Returns a dict: {building_type: DataFrame with columns ['ds', 'y']}
    """
    series_by_type = {}

    for building_type, group in df.groupby("primary_use"):
        if len(group) < MIN_ROWS_PER_TYPE:
            print(f"Skipping '{building_type}' — only {len(group)} rows (need {MIN_ROWS_PER_TYPE}+)")
            continue

        hourly_avg = (
            group.groupby("timestamp")["meter_reading"]
            .mean()
            .reset_index()
            .rename(columns={"timestamp": "ds", "meter_reading": "y"})
        )
        hourly_avg["ds"] = pd.to_datetime(hourly_avg["ds"])
        hourly_avg = hourly_avg.sort_values("ds")

        series_by_type[building_type] = hourly_avg
        print(f"'{building_type}': {len(hourly_avg):,} hourly points "
              f"({hourly_avg['ds'].min()} to {hourly_avg['ds'].max()})")

    return series_by_type


def train_prophet_models(series_by_type):
    models = {}

    for building_type, series in series_by_type.items():
        print(f"\nTraining Prophet model for '{building_type}'...")
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # dataset likely spans <1 year, avoid overfitting
        )
        model.fit(series)
        models[building_type] = model
        print(f"  Done.")

    return models


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows across {df['primary_use'].nunique()} building types")

    print("\nBuilding aggregated hourly series per building type...")
    series_by_type = build_hourly_series_per_type(df)

    if not series_by_type:
        raise ValueError(
            "No building types had enough data to train on. "
            "Check data/training_ready.csv and MIN_ROWS_PER_TYPE."
        )

    print(f"\n{len(series_by_type)} building types will get a trained model.")

    models = train_prophet_models(series_by_type)

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "models": models,
        "building_types": sorted(models.keys()),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved {len(models)} Prophet models to: {MODEL_PATH}")
    print(f"Building types covered: {artifact['building_types']}")


if __name__ == "__main__":
    main()