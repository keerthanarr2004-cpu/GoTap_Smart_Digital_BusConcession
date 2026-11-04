"""
Train and Evaluate Bus Demand Predictor
======================================
Uses Scikit-learn RandomForestRegressor to estimate future passenger demand
per bus route based on historical route usage records, temporal variables,
and student transit patterns.

Outputs:
- Saved Joblib pipeline: models/saved/demand_predictor.joblib
- Evaluation metrics: models/metrics/demand_predictor_metrics.json
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bus_demand_clean.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
METRICS_DIR = os.path.join(BASE_DIR, "models", "metrics")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def train_demand_predictor():
    print("Loading historical bus demand dataset...")
    df = pd.read_csv(DATA_PATH)
    
    cat_features = ['route_id', 'day_of_week', 'hour']
    num_features = ['is_weekend', 'is_peak_hour', 'is_monsoon', 'active_pass_holders', 'is_morning_peak', 'is_evening_peak']
    
    X = df[cat_features + num_features]
    y = df['passenger_demand']

    # Route metadata for lookups
    route_meta = df[['route_id', 'route_name', 'start_point', 'end_point']].drop_duplicates().to_dict('records')

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', StandardScaler(), num_features)
        ]
    )

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf_reg)
    ])

    print("Training RandomForestRegressor for Bus Demand Prediction...")
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    # Cross validation on sample for fast robust estimate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sample_indices = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
    cv_scores = cross_val_score(pipeline, X.iloc[sample_indices], y.iloc[sample_indices], cv=kf, scoring='r2')

    metrics = {
        "model_name": "GoTap Bus Demand Predictor",
        "algorithm": "RandomForestRegressor",
        "n_estimators": 100,
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "mape_percent": round(float(mape), 2),
        "cv_mean_r2": round(float(cv_scores.mean()), 4),
        "cv_std_r2": round(float(cv_scores.std()), 4),
    }

    print("\n--- Bus Demand Predictor Evaluation Results ---")
    print(f"Mean Absolute Error (MAE):     {metrics['mae']} passengers")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']} passengers")
    print(f"R-squared (R2 Score):          {metrics['r2_score']}")
    print(f"MAPE:                          {metrics['mape_percent']}%")
    print(f"5-Fold CV R2 Score:            {metrics['cv_mean_r2']} (+/- {metrics['cv_std_r2']})")

    # Save artifact bundle with Joblib
    artifact = {
        'pipeline': pipeline,
        'cat_features': cat_features,
        'num_features': num_features,
        'route_meta': route_meta,
        'metrics': metrics
    }

    model_save_path = os.path.join(SAVED_MODELS_DIR, "demand_predictor.joblib")
    joblib.dump(artifact, model_save_path, compress=3)
    print(f"Model artifact saved to: {model_save_path}")

    metrics_save_path = os.path.join(METRICS_DIR, "demand_predictor_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_save_path}")

    return artifact, metrics


if __name__ == "__main__":
    train_demand_predictor()
