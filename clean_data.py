"""
clean_data.py
Cleans the merged ASHRAE dataset (data/clean_training_data.csv) so it's
ready for model training:
  - drops columns that are too sparse to be useful (year_built, floor_count, cloud_coverage)
  - fills remaining missing values sensibly (0 for precip, median for weather)
  - drops rows with meter_reading == 0 (near-certain sensor outages)
  - extracts hour-of-day and is_weekend from the timestamp (useful signal for the model)

Run once:
    python clean_data.py

Output:
    data/training_ready.csv   <- final dataset, ready for train_model.py
"""

import pandas as pd
import numpy as np

INPUT_PATH = "data/clean_training_data.csv"
OUTPUT_PATH = "data/training_ready.csv"

COLUMNS_TO_DROP = ["year_built", "floor_count", "cloud_coverage"]
FILL_ZERO_COLUMNS = ["precip_depth_1_hr"]
FILL_MEDIAN_COLUMNS = [
    "sea_level_pressure",
    "wind_direction",
    "air_temperature",
    "dew_temperature",
    "wind_speed",
]


def clean_dataset():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Starting rows: {len(df):,}")

    # Drop sparse columns
    print(f"\nDropping sparse columns: {COLUMNS_TO_DROP}")
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Fill precipitation gaps with 0 (missing = no precipitation recorded)
    for col in FILL_ZERO_COLUMNS:
        n_filled = df[col].isnull().sum()
        df[col] = df[col].fillna(0)
        print(f"Filled {n_filled:,} missing values in '{col}' with 0")

    # Fill remaining weather gaps with the column median
    for col in FILL_MEDIAN_COLUMNS:
        n_filled = df[col].isnull().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {n_filled:,} missing values in '{col}' with median ({median_val:.2f})")

    # Drop rows where meter_reading == 0 (sensor outages, not real usage)
    n_before = len(df)
    df = df[df["meter_reading"] > 0]
    n_dropped = n_before - len(df)
    print(f"\nDropped {n_dropped:,} rows with meter_reading == 0 (sensor outages)")

    # Extract useful time-based features from the timestamp
    print("\nExtracting hour-of-day and is_weekend from timestamp...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    # Sanity check: no nulls should remain anywhere
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"\nWARNING: {remaining_nulls} null values still remain — check output above.")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("\nNo missing values remain. Good to go.")

    print(f"\nFinal rows: {len(df):,}")
    print(f"Final columns: {list(df.columns)}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")

    import os
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    clean_dataset()