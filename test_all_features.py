"""
Unit and Integration Tests for GoTap AI/ML Features
===================================================
Tests all 4 AI/ML modules and Django endpoints:
1. ML Route Recommendation & API
2. NLP Feedback & Complaint Sentiment + Issue Categorization
3. Bus Demand Prediction Dashboard & Service
4. Conversational Student AI Assistant & API
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import django
from django.test import TestCase, Client

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "concession.settings")
django.setup()

from stc.models import BusRoute, Feedback, Complaints, user
from stc.ml_services import (
    recommend_routes,
    find_best_route,
    analyze_sentiment,
    classify_issue_category,
    BusDemandPredictor,
    answer_student_question
)


def run_tests():
    print("=" * 60)
    print("      GOTAP AI/ML SYSTEM VERIFICATION TESTS")
    print("=" * 60)
    client = Client()

    # --- Test 1: ML Route Recommendation ---
    print("\n[Test 1] Testing ML Route Recommendation Engine...")
    recs = recommend_routes("Kerala Varma College", "Ollur", top_n=3)
    assert len(recs) > 0, "No route recommendations returned"
    top_rec = recs[0]
    print(f"  [OK] Top Recommendation: {top_rec['start_point']} -> {top_rec['end_point']} (Confidence: {top_rec['confidence']}%, Model: {top_rec.get('model_source')})")
    assert top_rec['id'] == 158 or top_rec['id'] == 180, "Unexpected route ID recommended"

    # API Route Test
    r_api = client.get('/api/recommend-routes/?from=Kerala+Varma+College&to=Ollur')
    assert r_api.status_code == 200, f"Expected 200, got {r_api.status_code}"
    assert len(r_api.json()['recommendations']) > 0, "API returned empty recommendations"
    print("  [OK] GET /api/recommend-routes/ HTTP 200 OK")

    # --- Test 2: Bus Demand Prediction ---
    print("\n[Test 2] Testing Bus Demand Prediction Engine...")
    route = BusRoute.objects.first()
    assert route is not None, "No routes in DB"
    pred = BusDemandPredictor.predict(route.id, day_of_week=0, hour=8)
    assert 'predicted_passengers' in pred, "Missing predicted_passengers"
    assert 'demand_level' in pred, "Missing demand_level"
    print(f"  [OK] Predicted Rush: {pred['predicted_passengers']} passengers (Level: {pred['demand_level']}, Fleet Advice: '{pred['recommendation'][:40]}...')")

    # Demand Forecast View
    r_demand = client.get('/demand_forecast/')
    assert r_demand.status_code == 200, f"Expected 200, got {r_demand.status_code}"
    print("  [OK] GET /demand_forecast/ HTTP 200 OK")

    # --- Test 3: NLP Sentiment & Multi-Class Issue Categorization ---
    print("\n[Test 3] Testing NLP Feedback & Complaint Analysis...")
    sample_texts = [
        ("Conductor was shouting at students and refused the pass.", "Negative", "Conductor & Staff Behavior"),
        ("Bus was 30 minutes late today at Sakthan bus stand.", "Negative", "Bus Delay / Punctuality"),
        ("Extremely crowded bus, standing on footboard.", "Negative", "Overcrowding & Rush"),
        ("GoTap digital concession system has made student bus travel so convenient and hassle-free!", "Positive", "Appreciation / Service Commendation")
    ]
    for text, expected_sent, expected_cat in sample_texts:
        res = analyze_sentiment(text)
        print(f"  [OK] Text: \"{text[:35]}...\" -> Sentiment: {res['sentiment']} ({res['score']}), Category: {res['issue_category']}")
        assert res['sentiment'] == expected_sent, f"Sentiment mismatch: {res['sentiment']} != {expected_sent}"
        assert res['issue_category'] == expected_cat, f"Category mismatch: {res['issue_category']} != {expected_cat}"

    # Feedback list & complaint list views
    r_fb = client.get('/feedbacklist/')
    assert r_fb.status_code == 200, f"Expected 200, got {r_fb.status_code}"
    print("  [OK] GET /feedbacklist/ HTTP 200 OK")

    r_cp = client.get('/complaintlist/')
    assert r_cp.status_code == 200, f"Expected 200, got {r_cp.status_code}"
    print("  [OK] GET /complaintlist/ HTTP 200 OK")

    # --- Test 4: Student AI Assistant ---
    print("\n[Test 4] Testing Conversational Student AI Assistant...")
    queries = [
        ("How do I apply for a bus pass?", "bus_pass_apply"),
        ("How do QR codes work on the bus?", "qr_code_usage"),
        ("How can I recharge my wallet balance?", "wallet_and_payment"),
        ("How to report a rude conductor?", "complaint_and_feedback")
    ]
    for q, expected_intent in queries:
        ans = answer_student_question(q)
        print(f"  [OK] Query: \"{q}\" -> Intent: {ans['intent']} (Confidence: {ans['confidence']}%)")
        assert ans['intent'] == expected_intent, f"Intent mismatch: {ans['intent']} != {expected_intent}"

    # Assistant view & JSON API
    r_ast = client.get('/assistant/')
    assert r_ast.status_code == 200, f"Expected 200, got {r_ast.status_code}"
    print("  [OK] GET /assistant/ HTTP 200 OK")

    r_ast_api = client.get('/api/student-assistant/?query=how+to+apply')
    assert r_ast_api.status_code == 200, f"Expected 200, got {r_ast_api.status_code}"
    assert r_ast_api.json()['intent'] == 'bus_pass_apply'
    print("  [OK] GET /api/student-assistant/ HTTP 200 OK")

    print("\n" + "=" * 60)
    print("  ALL AI/ML FEATURES TESTED & VALIDATED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
