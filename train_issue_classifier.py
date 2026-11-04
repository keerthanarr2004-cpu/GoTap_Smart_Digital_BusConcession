"""
Train and Evaluate NLP Feedback and Complaint Issue Classifier
=============================================================
Uses TF-IDF Vectorization and LogisticRegression / ComplementNB to classify student
feedback and complaints into 7 operational categories:
1. Bus Delay / Punctuality
2. Overcrowding & Rush
3. Conductor & Staff Behavior
4. Pass, Payment & QR Scan Issue
5. Cleanliness & Maintenance
6. Route & Stop Coverage
7. Appreciation / Service Commendation

Outputs:
- Saved Joblib pipeline: models/saved/issue_classifier.joblib
- Evaluation metrics: models/metrics/issue_classifier_metrics.json
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feedback_complaint_clean.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
METRICS_DIR = os.path.join(BASE_DIR, "models", "metrics")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def train_issue_classifier():
    print("Loading feedback/complaint NLP dataset...")
    df = pd.read_csv(DATA_PATH)
    
    X = df['clean_text']
    y = df['issue_category']

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            strip_accents='unicode'
        )),
        ('classifier', LogisticRegression(
            C=2.0,
            max_iter=500,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    print("Training NLP Issue Classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    categories = list(pipeline.classes_)

    metrics = {
        "model_name": "GoTap Transit Issue NLP Classifier",
        "algorithm": "TF-IDF + LogisticRegression",
        "categories_count": len(categories),
        "categories": categories,
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train),
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_score_weighted": round(float(f1), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
    }

    print("\n--- NLP Issue Classifier Evaluation Results ---")
    print(f"Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"Weighted Precision: {metrics['precision_weighted']*100:.2f}%")
    print(f"Weighted Recall:    {metrics['recall_weighted']*100:.2f}%")
    print(f"Weighted F1 Score:  {metrics['f1_score_weighted']*100:.2f}%")
    print(f"5-Fold CV Accuracy: {metrics['cv_mean_accuracy']*100:.2f}% (+/- {metrics['cv_std_accuracy']*100:.2f}%)")

    # Save artifact
    artifact = {
        'pipeline': pipeline,
        'categories': categories,
        'metrics': metrics
    }

    model_save_path = os.path.join(SAVED_MODELS_DIR, "issue_classifier.joblib")
    joblib.dump(artifact, model_save_path, compress=3)
    print(f"Model artifact saved to: {model_save_path}")

    metrics_save_path = os.path.join(METRICS_DIR, "issue_classifier_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_save_path}")

    return artifact, metrics


if __name__ == "__main__":
    train_issue_classifier()
