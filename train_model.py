"""
train_model.py
Trains a RandomForestRegressor on the cleaned ASHRAE dataset
(data/training_ready.csv) to predict electricity meter_reading (kWh).

Run once (or whenever you want to retrain):
    python train_model.py

Output:
    models/energy_model.pkl   <- trained model + metadata, loaded by utils/ml_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_PATH = "data/training_ready.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "energy_model.pkl")
RANDOM_SEED = 42

# Categorical column that needs encoding
CATEGORICAL_COL = "primary_use"

# Numeric features the model will actually use
NUMERIC_FEATURES = [
    "square_feet",
    "air_temperature",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "hour",
    "is_weekend",
]

TARGET_COL = "meter_reading"


def load_and_prepare_data():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")

    # One-hot encode primary_use (e.g. "Office", "Education" -> separate 0/1 columns)
    print(f"\nEncoding categorical column '{CATEGORICAL_COL}'...")
    dummies = pd.get_dummies(df[CATEGORICAL_COL], prefix="use")
    dummy_cols = list(dummies.columns)
    print(f"Created {len(dummy_cols)} category columns: {dummy_cols}")

    df = pd.concat([df, dummies], axis=1)

    feature_cols = NUMERIC_FEATURES + dummy_cols
    X = df[feature_cols]
    y = df[TARGET_COL]

    return X, y, feature_cols


def train_and_save_model():
    X, y, feature_cols = load_and_prepare_data()

    print(f"\nSplitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    print("\nTraining RandomForestRegressor (this may take a few minutes on ~2M rows)...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out test set...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\nModel performance:")
    print(f"  MAE : {mae:.2f} kWh")
    print(f"  R^2 : {r2:.3f}")

    # Feature importance (useful sanity check — square_feet and temperature
    # should usually dominate)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))

    os.makedirs(MODEL_DIR, exist_ok=True)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_col": CATEGORICAL_COL,
        "category_values": sorted(
            [c.replace("use_", "") for c in feature_cols if c.startswith("use_")]
        ),
        "mae": mae,
        "r2": r2,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()