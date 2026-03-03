"""
model.py  —  ML Layer
=====================
Responsible for:
  - Feature engineering  (creating new columns from existing ones)
  - Training the model   (if no saved model exists)
  - Loading the model    (from the .pkl file)
  - Making predictions   (returning score + label)
  - Generating tips      (business logic based on input values)

This file has ZERO Streamlit code.
It can be imported by any frontend: Streamlit, Flask, FastAPI, Telegram bot, etc.
"""

import os
import pandas as pd
import numpy as np
import joblib

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_PATH   = "wine_quality_model_v2.pkl"
CSV_PATH     = "winequality-white.csv"
CSV_PATH_ALT = "winequality_white.csv"

# ── Reference statistics (computed from the full UCI dataset) ─────────────────
DATASET_AVERAGES = {
    "fixed acidity":       6.85,
    "volatile acidity":    0.278,
    "citric acid":         0.334,
    "residual sugar":      6.39,
    "chlorides":           0.046,
    "free sulfur dioxide": 35.3,
    "total sulfur dioxide":138.4,
    "density":             0.994,
    "pH":                  3.19,
    "sulphates":           0.490,
    "alcohol":             10.51,
}

PREMIUM_AVERAGES = {
    "fixed acidity":       6.73,
    "volatile acidity":    0.262,
    "citric acid":         0.337,
    "residual sugar":      5.18,
    "chlorides":           0.043,
    "free sulfur dioxide": 34.0,
    "total sulfur dioxide":125.1,
    "density":             0.992,
    "pH":                  3.21,
    "sulphates":           0.50,
    "alcohol":             11.37,
}


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4 new domain-knowledge features to the dataframe.
    Must be applied consistently during both training and prediction.

    New features:
      free_sulfur_ratio   — proportion of free vs total SO2 (preservation quality)
      acidity_balance     — ratio of fixed to volatile acidity
      sugar_alcohol_ratio — sweetness vs alcohol level
      total_acidity       — sum of all acidity measures
    """
    df = df.copy()
    df["free_sulfur_ratio"]   = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-5)
    df["acidity_balance"]     = df["fixed acidity"]       / (df["volatile acidity"]     + 1e-5)
    df["sugar_alcohol_ratio"] = df["residual sugar"]      / (df["alcohol"]              + 1e-5)
    df["total_acidity"]       = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    return df


# ── Model Training ────────────────────────────────────────────────────────────

def train_and_save_model() -> object:
    """
    Train a Random Forest pipeline on the wine CSV and save it to disk.
    Called automatically when no saved model is found.

    Returns the trained pipeline.
    Raises FileNotFoundError if the CSV is not found in either expected location.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Find the CSV
    csv_file = None
    for path in [CSV_PATH, CSV_PATH_ALT]:
        if os.path.exists(path):
            csv_file = path
            break

    if csv_file is None:
        raise FileNotFoundError(
            f"Dataset not found. Place '{CSV_PATH}' in the project folder."
        )

    # Load and clean
    df = pd.read_csv(csv_file, sep=";")
    df = df.drop_duplicates()

    # Feature engineering
    df = engineer_features(df)

    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Build pipeline: StandardScaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model() -> object:
    """
    Load the saved model from disk.
    If no saved model exists, train a fresh one first.

    Returns the trained sklearn Pipeline.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # No saved model — train from scratch
    return train_and_save_model()


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, inputs: dict) -> dict:
    """
    Run a prediction for one wine sample.

    Parameters
    ----------
    model  : trained sklearn Pipeline (from load_model())
    inputs : dict with all 11 raw wine features, e.g.
             {"fixed acidity": 7.0, "volatile acidity": 0.28, ...}

    Returns
    -------
    dict with:
        score        (float)  — predicted quality 0–10
        label        (str)    — "Premium Quality" / "Normal Quality" / "Low Quality"
        badge_class  (str)    — CSS class name for the frontend
        emoji        (str)    — matching emoji
    """
    input_df     = pd.DataFrame([inputs])
    input_df     = engineer_features(input_df)
    raw_score    = float(model.predict(input_df)[0])
    score        = round(min(max(raw_score, 0), 10), 2)

    if score >= 7:
        label, badge_class, emoji = "Premium Quality", "badge-premium", "🏆"
    elif score >= 5:
        label, badge_class, emoji = "Normal Quality",  "badge-normal",  "✅"
    else:
        label, badge_class, emoji = "Low Quality",     "badge-low",     "⚠️"

    return {
        "score":       score,
        "label":       label,
        "badge_class": badge_class,
        "emoji":       emoji,
    }


# ── Smart Tips (Business Logic) ───────────────────────────────────────────────

def generate_tips(inputs: dict) -> dict:
    """
    Analyse wine inputs and return domain-knowledge tips.

    Returns
    -------
    dict with:
        tips     (list of str) — things the wine does well
        warnings (list of str) — things to improve
    """
    tips     = []
    warnings = []

    # ── Warnings ─────────────────────────────────────────────────────────────
    if inputs["volatile acidity"] > 0.35:
        warnings.append(
            "⚠️ **Volatile acidity is high** — gives wine a vinegar taste. "
            "Keep below 0.30 for premium quality."
        )
    if inputs["alcohol"] < 10.0:
        warnings.append(
            "⚠️ **Alcohol is low** — alcohol is the #1 quality driver. "
            "Premium wines average 11%+."
        )
    if inputs["residual sugar"] > 12:
        warnings.append(
            "⚠️ **Residual sugar is very high** — may make wine overly sweet "
            "and reduce the quality score."
        )
    if inputs["chlorides"] > 0.07:
        warnings.append(
            "⚠️ **Chlorides are elevated** — high salt content negatively impacts taste."
        )

    # ── Strengths ─────────────────────────────────────────────────────────────
    if inputs["alcohol"] >= 11.0:
        tips.append("✅ **Good alcohol level** — within the premium wine range.")
    if inputs["volatile acidity"] <= 0.28:
        tips.append("✅ **Volatile acidity well-controlled** — minimal vinegar taste expected.")
    if 20 < inputs["free sulfur dioxide"] < 60:
        tips.append("✅ **SO₂ levels balanced** — good preservation without excess.")
    if 3.0 <= inputs["pH"] <= 3.4:
        tips.append("✅ **pH in ideal range** — good acidity balance for white wine.")

    return {"tips": tips, "warnings": warnings}
