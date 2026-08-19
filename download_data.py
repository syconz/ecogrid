"""
download_data.py
Downloads the ASHRAE Great Energy Predictor III competition dataset via the
Kaggle API, then immediately filters it down to something manageable:
  - electricity meters only (meter == 0)
  - a random subsample of buildings (not all 2,380)
  - merged with building metadata and weather data into one clean CSV

Run once:
    python download_data.py

Output:
    data/raw/                    <- original downloaded Kaggle files (kept, but not used again)
    data/clean_training_data.csv <- the merged, filtered dataset the next script will train on
"""

import os
import zipfile
import pandas as pd
import numpy as np

COMPETITION = "ashrae-energy-prediction"
RAW_DIR = "data/raw"
CLEAN_CSV_PATH = "data/clean_training_data.csv"

N_BUILDINGS_TO_SAMPLE = 250
RANDOM_SEED = 42


def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)

    required_files = ["train.csv", "building_metadata.csv", "weather_train.csv"]
    already_have_all = all(
        os.path.exists(os.path.join(RAW_DIR, f)) for f in required_files
    )

    if already_have_all:
        print("Raw files already present in data/raw/, skipping download.")
        return

    print(f"Downloading competition files for '{COMPETITION}' via Kaggle API...")
    print("(This can take a few minutes depending on your connection — the full")
    print(" competition zip is a couple GB even though we'll filter it down after.)")

    exit_code = os.system(
        f'kaggle competitions download -c {COMPETITION} -p "{RAW_DIR}"'
    )
    if exit_code != 0:
        raise RuntimeError(
            "Kaggle download failed. Common causes:\n"
            "  1. You haven't accepted the competition rules at "
            f"https://www.kaggle.com/c/{COMPETITION}/rules\n"
            "  2. kaggle.json credentials are missing/invalid\n"
            "Fix the issue above and re-run this script."
        )

    zip_path = os.path.join(RAW_DIR, f"{COMPETITION}.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Expected zip file not found at {zip_path} after download."
        )

    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    print("Extraction complete.")


def build_clean_dataset():
    print("\nLoading building metadata...")
    buildings = pd.read_csv(os.path.join(RAW_DIR, "building_metadata.csv"))

    print("Loading weather data...")
    weather = pd.read_csv(os.path.join(RAW_DIR, "weather_train.csv"))

    # Pick a random subsample of building_ids to keep the dataset small
    rng = np.random.default_rng(RANDOM_SEED)
    all_building_ids = buildings["building_id"].unique()
    sampled_ids = rng.choice(
        all_building_ids,
        size=min(N_BUILDINGS_TO_SAMPLE, len(all_building_ids)),
        replace=False,
    )
    print(f"Sampled {len(sampled_ids)} buildings out of {len(all_building_ids)} total.")

    print("\nReading train.csv in chunks and filtering "
          "(electricity meter + sampled buildings only)...")
    filtered_chunks = []
    chunksize = 2_000_000
    sampled_ids_set = set(sampled_ids)

    reader = pd.read_csv(
        os.path.join(RAW_DIR, "train.csv"),
        chunksize=chunksize,
        usecols=["building_id", "meter", "timestamp", "meter_reading"],
    )

    for i, chunk in enumerate(reader):
        chunk = chunk[
            (chunk["meter"] == 0) & (chunk["building_id"].isin(sampled_ids_set))
        ]
        filtered_chunks.append(chunk)
        print(f"  Processed chunk {i + 1} — kept {len(chunk):,} rows so far this chunk")

    train_filtered = pd.concat(filtered_chunks, ignore_index=True)
    print(f"\nTotal filtered readings: {len(train_filtered):,} rows")

    if len(train_filtered) == 0:
        raise ValueError(
            "No rows survived filtering. This usually means train.csv wasn't "
            "found/extracted correctly — check data/raw/ for train.csv."
        )

    # Merge in building metadata (adds site_id, primary_use, square_feet, etc.)
    merged = train_filtered.merge(buildings, on="building_id", how="left")

    # Merge in weather data (matched on site_id + timestamp)
    merged = merged.merge(weather, on=["site_id", "timestamp"], how="left")

    print(f"After merging metadata + weather: {len(merged):,} rows, "
          f"{merged.shape[1]} columns")

    os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)
    merged.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"\nSaved clean merged dataset to: {CLEAN_CSV_PATH}")
    print(f"File size: {os.path.getsize(CLEAN_CSV_PATH) / (1024*1024):.1f} MB")

    print("\nPreview of columns:")
    print(list(merged.columns))
    print("\nPreview of first few rows:")
    print(merged.head())


if __name__ == "__main__":
    download_and_extract()
    build_clean_dataset()
