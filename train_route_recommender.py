"""
Train and Evaluate ML Route Recommendation Model
================================================
Uses Scikit-learn RandomForestClassifier to recommend the optimal bus route
based on student journey attributes, origin, destination, distance, fare, and college.

Outputs:
- Saved Joblib pipeline: models/saved/route_recommender.joblib
- Evaluation metrics: models/metrics/route_recommender_metrics.json
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "route_travel_clean.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
METRICS_DIR = os.path.join(BASE_DIR, "models", "metrics")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def train_route_recommender():
    print("Loading route recommendation dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Feature columns
    cat_features = ['origin_stop_clean', 'destination_stop_clean', 'college']
    num_features = ['distance_km', 'fare', 'travel_hour', 'day_of_week']
    
    X = df[cat_features + num_features]
    y = df['target_route_id']
    
    # Route mapping metadata for rapid inference lookup
    route_meta = df[['target_route_id', 'target_route_name', 'canonical_origin', 'canonical_destination', 'fare']].drop_duplicates()
    route_dict = {}
    for _, row in route_meta.iterrows():
        route_dict[int(row['target_route_id'])] = {
            'route_id': int(row['target_route_id']),
            'route_name': str(row['target_route_name']),
            'start_point': str(row['canonical_origin']),
            'end_point': str(row['canonical_destination']),
            'fare': float(row['fare'])
        }

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', StandardScaler(), num_features)
        ]
    )

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])

    print("Training RandomForestClassifier for Route Recommendation...")
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    
    # Top-3 Accuracy
    classes = pipeline.classes_
    top3_correct = 0
    for true_label, probas in zip(y_test, y_pred_proba):
        top3_indices = np.argsort(probas)[-3:][::-1]
        top3_classes = classes[top3_indices]
        if true_label in top3_classes:
            top3_correct += 1
    top3_acc = top3_correct / len(y_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # 5-fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    metrics = {
        "model_name": "Random Forest Route Recommender",
        "algorithm": "RandomForestClassifier",
        "n_estimators": 120,
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train),
        "accuracy": round(float(acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_score_weighted": round(float(f1), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
    }

    print("\n--- Route Recommender Evaluation Results ---")
    print(f"Top-1 Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Top-3 Accuracy:     {metrics['top3_accuracy']*100:.2f}%")
    print(f"Weighted Precision: {metrics['precision_weighted']*100:.2f}%")
    print(f"Weighted Recall:    {metrics['recall_weighted']*100:.2f}%")
    print(f"Weighted F1 Score:  {metrics['f1_score_weighted']*100:.2f}%")
    print(f"5-Fold CV Accuracy: {metrics['cv_mean_accuracy']*100:.2f}% (+/- {metrics['cv_std_accuracy']*100:.2f}%)")

    # Save artifact bundle with Joblib
    artifact = {
        'pipeline': pipeline,
        'cat_features': cat_features,
        'num_features': num_features,
        'route_dict': route_dict,
        'classes': list(classes),
        'metrics': metrics
    }
    
    model_save_path = os.path.join(SAVED_MODELS_DIR, "route_recommender.joblib")
    joblib.dump(artifact, model_save_path, compress=3)
    print(f"Model artifact saved to: {model_save_path}")

    metrics_save_path = os.path.join(METRICS_DIR, "route_recommender_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_save_path}")

    return artifact, metrics


if __name__ == "__main__":
    train_route_recommender()
