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



# =============================================================================
# MODEL EVALUATION  -- NEW SECTION
# =============================================================================

def compute_model_metrics() -> dict:
    """
    Train Random Forest AND Linear Regression on 80/20 train-test split.
    Compute R2, MAE, RMSE, and 5-fold cross-validation for both.

    Why each metric matters
    -----------------------
    R2   : How much wine quality variation the model explains.
           Range 0-1. R2=0.55 -> model explains 55% of quality differences.
           Higher is better.

    MAE  : Mean Absolute Error. Average error in quality points.
           MAE=0.48 -> predictions off by 0.48 points on average (0-10 scale).
           Lower is better.

    RMSE : Root Mean Squared Error. Like MAE but penalises large errors more.
           If RMSE much bigger than MAE -> occasional very wrong predictions.
           Lower is better.

    CV   : 5-fold Cross-Validation. Data split 5 ways, model trained & tested
           5 separate times on different splits. Consistent CV scores prove
           the model is reliable -- not just lucky on one test set.

    We also compute Linear Regression metrics (the original basic model) so
    the viewer can clearly see WHY we upgraded to Random Forest.

    Called once by the Streamlit frontend (result is cached with st.cache_data).

    Returns
    -------
    dict with:
        r2, mae, rmse          -- Random Forest test-set metrics
        cv_scores              -- numpy array: all 5 individual fold R2 scores
        cv_mean, cv_std        -- mean and std-dev of CV scores
        lr_r2, lr_mae, lr_rmse -- Linear Regression baseline metrics
        n_train, n_test        -- number of samples in each split
        n_features             -- total features used (11 original + 4 engineered)
    Returns None if the CSV is not found.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Find the dataset file
    csv_file = None
    for path in [CSV_PATH, CSV_PATH_ALT]:
        if os.path.exists(path):
            csv_file = path
            break
    if csv_file is None:
        return None

    # Prepare data (same steps as training)
    df = pd.read_csv(csv_file, sep=";").drop_duplicates()
    df = engineer_features(df)
    X  = df.drop("quality", axis=1)
    y  = df["quality"]

    # 80/20 split -- same random seed so results are reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Random Forest (our chosen model) --------------------------------
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred    = rf_pipe.predict(X_test)
    r2        = r2_score(y_test, y_pred)
    mae       = mean_absolute_error(y_test, y_pred)
    rmse      = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    cv_scores = cross_val_score(rf_pipe, X, y, cv=5, scoring="r2")

    # ---- Linear Regression (baseline -- original project used this) ------
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)

    return {
        "r2":         r2,
        "mae":        mae,
        "rmse":       rmse,
        "cv_scores":  cv_scores,
        "cv_mean":    float(cv_scores.mean()),
        "cv_std":     float(cv_scores.std()),
        "lr_r2":      r2_score(y_test, lr_pred),
        "lr_mae":     mean_absolute_error(y_test, lr_pred),
        "lr_rmse":    float(np.sqrt(mean_squared_error(y_test, lr_pred))),
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_features": X.shape[1],
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
