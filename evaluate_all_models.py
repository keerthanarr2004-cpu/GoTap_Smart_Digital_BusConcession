"""
Comprehensive Model Evaluation & Verification Suite for GoTap AI/ML
===================================================================
Evaluates and benchmarks all 3 GoTap AI/ML models:
1. ML Route Recommender (RandomForestClassifier)
2. NLP Issue Classifier (TF-IDF + LogisticRegression)
3. Bus Demand Predictor (RandomForestRegressor)
Also verifies VADER sentiment analysis accuracy and compiles models/metrics/model_summary.json.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
METRICS_DIR = os.path.join(BASE_DIR, "models", "metrics")


def evaluate_all():
    print("=" * 70)
    print("      GOTAP SMART TRANSPORTATION SYSTEM: AI/ML MODEL EVALUATION")
    print("=" * 70)

    # 1. Route Recommender
    rr_path = os.path.join(SAVED_DIR, "route_recommender.joblib")
    assert os.path.exists(rr_path), f"Missing {rr_path}"
    rr_bundle = joblib.load(rr_path)
    rr_pipeline = rr_bundle['pipeline']
    rr_metrics = rr_bundle['metrics']
    
    # Test inference speed & test sample
    t0 = time.time()
    sample_trip = pd.DataFrame([{
        'origin_stop_clean': 'Kerala Varma College',
        'destination_stop_clean': 'Ollur',
        'college': 'Sree Kerala Varma College',
        'distance_km': 6.8,
        'fare': 5.0,
        'travel_hour': 8,
        'day_of_week': 0
    }])
    pred_route = rr_pipeline.predict(sample_trip)[0]
    t_rr = (time.time() - t0) * 1000
    
    print("\n1. ROUTE RECOMMENDATION MODEL:")
    print(f"   - Model: {rr_metrics['algorithm']} ({rr_metrics['n_estimators']} trees)")
    print(f"   - Top-1 Accuracy:     {rr_metrics['accuracy']*100:.2f}%")
    print(f"   - Top-3 Accuracy:     {rr_metrics['top3_accuracy']*100:.2f}%")
    print(f"   - Weighted F1 Score:  {rr_metrics['f1_score_weighted']*100:.2f}%")
    print(f"   - 5-Fold CV Accuracy: {rr_metrics['cv_mean_accuracy']*100:.2f}%")
    print(f"   - Sample Trip Prediction: Route ID {pred_route} (Inference: {t_rr:.2f} ms)")

    # 2. NLP Issue Classifier
    ic_path = os.path.join(SAVED_DIR, "issue_classifier.joblib")
    assert os.path.exists(ic_path), f"Missing {ic_path}"
    ic_bundle = joblib.load(ic_path)
    ic_pipeline = ic_bundle['pipeline']
    ic_metrics = ic_bundle['metrics']

    t0 = time.time()
    sample_text = ["Conductor shouted at students and refused digital QR pass at Sakthan stand."]
    pred_category = ic_pipeline.predict(sample_text)[0]
    t_ic = (time.time() - t0) * 1000

    print("\n2. NLP FEEDBACK & COMPLAINT CLASSIFIER:")
    print(f"   - Model: {ic_metrics['algorithm']}")
    print(f"   - Categories ({ic_metrics['categories_count']}): {', '.join(ic_metrics['categories'][:3])}...")
    print(f"   - Test Accuracy:     {ic_metrics['accuracy']*100:.2f}%")
    print(f"   - Weighted F1 Score: {ic_metrics['f1_score_weighted']*100:.2f}%")
    print(f"   - Sample Complaint Category: '{pred_category}' (Inference: {t_ic:.2f} ms)")

    # 3. Bus Demand Predictor
    dp_path = os.path.join(SAVED_DIR, "demand_predictor.joblib")
    assert os.path.exists(dp_path), f"Missing {dp_path}"
    dp_bundle = joblib.load(dp_path)
    dp_pipeline = dp_bundle['pipeline']
    dp_metrics = dp_bundle['metrics']

    t0 = time.time()
    sample_demand_input = pd.DataFrame([{
        'route_id': 158,
        'day_of_week': 0,
        'hour': 8,
        'is_weekend': 0,
        'is_peak_hour': 1,
        'is_monsoon': 0,
        'active_pass_holders': 45,
        'is_morning_peak': 1,
        'is_evening_peak': 0
    }])
    pred_demand = dp_pipeline.predict(sample_demand_input)[0]
    t_dp = (time.time() - t0) * 1000

    print("\n3. BUS DEMAND PREDICTION MODEL:")
    print(f"   - Model: {dp_metrics['algorithm']} ({dp_metrics['n_estimators']} trees)")
    print(f"   - Mean Absolute Error (MAE):     {dp_metrics['mae']} passengers")
    print(f"   - Root Mean Squared Error (RMSE): {dp_metrics['rmse']} passengers")
    print(f"   - R-squared (R2 Score):          {dp_metrics['r2_score']}")
    print(f"   - MAPE:                          {dp_metrics['mape_percent']}%")
    print(f"   - Sample Peak Demand Prediction: {int(round(pred_demand))} passengers (Inference: {t_dp:.2f} ms)")

    # 4. VADER Sentiment Analysis Benchmark
    sia = SentimentIntensityAnalyzer()
    test_sentences = [
        ("The digital bus pass system is amazingly fast and convenient!", "Positive"),
        ("Bus was running on normal schedule today.", "Neutral"),
        ("Horrible delay and rude bus conductor. Very disappointed.", "Negative"),
    ]
    correct_sent = 0
    for text, expected in test_sentences:
        score = sia.polarity_scores(text)['compound']
        pred_sent = 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')
        if pred_sent == expected:
            correct_sent += 1
    sent_acc = correct_sent / len(test_sentences)
    print("\n4. NLP SENTIMENT ANALYZER (NLTK VADER):")
    print(f"   - Benchmark Accuracy on transit sentiment: {sent_acc*100:.1f}%")

    # Compile Summary
    summary = {
        "project": "GoTap - Smart Student Transportation System",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "route_recommender": rr_metrics,
            "issue_classifier": ic_metrics,
            "demand_predictor": dp_metrics,
            "sentiment_analyzer": {
                "engine": "NLTK VADER Lexicon",
                "accuracy": round(sent_acc, 4)
            }
        }
    }

    summary_file = os.path.join(METRICS_DIR, "model_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nModel summary successfully compiled to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_all()
