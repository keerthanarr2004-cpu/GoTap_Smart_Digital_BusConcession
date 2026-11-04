"""
Route Recommendation Service for GoTap
======================================
Loads the Joblib-persisted Random Forest Route Recommendation model
and provides high-speed route inference with calibrated match confidence,
stop normalization, and fuzzy string fallback.
"""

import os
import re
import difflib
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "route_recommender.joblib")

# Global cached artifact
_ROUTE_BUNDLE = None


def get_route_model():
    global _ROUTE_BUNDLE
    if _ROUTE_BUNDLE is None and os.path.exists(MODEL_PATH):
        try:
            _ROUTE_BUNDLE = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading route recommender model: {e}")
            _ROUTE_BUNDLE = None
    return _ROUTE_BUNDLE


def _normalize_location(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[\(\),.-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def calculate_similarity(a, b):
    norm_a = _normalize_location(a)
    norm_b = _normalize_location(b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    if norm_a in norm_b or norm_b in norm_a:
        return 0.9
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    seq_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
    return max(overlap, seq_ratio)


def recommend_routes(from_address, to_address, college="Sree Kerala Varma College", top_n=5):
    """
    Recommend best matching bus routes using the trained Random Forest model,
    supplemented by database fuzzy stop matching.
    """
    from stc.models import BusRoute
    
    routes = list(BusRoute.objects.all())
    if not routes:
        return []

    norm_from = _normalize_location(from_address)
    norm_to = _normalize_location(to_address)

    bundle = get_route_model()
    ml_recs = []
    
    # 1. ML Model Inference
    if bundle and bundle.get('pipeline') is not None:
        try:
            pipeline = bundle['pipeline']
            classes = bundle['classes']
            
            # Construct feature DataFrame for inference
            features = pd.DataFrame([{
                'origin_stop_clean': norm_from.title(),
                'destination_stop_clean': norm_to.title(),
                'college': college or "Sree Kerala Varma College",
                'distance_km': 10.0,
                'fare': 5.0,
                'travel_hour': 8,
                'day_of_week': 0
            }])
            
            probas = pipeline.predict_proba(features)[0]
            top_indices = np.argsort(probas)[::-1][:top_n]
            
            route_lookup = {r.id: r for r in routes}
            for idx in top_indices:
                route_id = classes[idx]
                prob = probas[idx]
                if route_id in route_lookup and prob > 0.02:
                    r = route_lookup[route_id]
                    # Score confidence blending model probability + text similarity
                    sim_start = calculate_similarity(from_address, r.start_point)
                    sim_end = calculate_similarity(to_address, r.end_point)
                    text_sim = (sim_start * 0.5 + sim_end * 0.5)
                    confidence = round(((prob * 0.5) + (text_sim * 0.5)) * 100, 1)
                    
                    is_exact = (
                        norm_from == _normalize_location(r.start_point) and
                        norm_to == _normalize_location(r.end_point)
                    )
                    
                    ml_recs.append({
                        'id': r.id,
                        'route_name': r.route_name,
                        'start_point': r.start_point,
                        'end_point': r.end_point,
                        'fare': float(r.fare),
                        'confidence': 100.0 if is_exact else max(confidence, round(text_sim * 100, 1)),
                        'is_exact': is_exact,
                        'is_return_direction': False,
                        'model_source': 'RandomForestClassifier'
                    })
        except Exception as e:
            print(f"ML route inference fallback to fuzzy matcher: {e}")

    # 2. If ML didn't produce enough recommendations, compute fuzzy matching over all routes
    existing_ids = {r['id'] for r in ml_recs}
    fuzzy_recs = []
    for route in routes:
        if route.id in existing_ids:
            continue
        start_sim = calculate_similarity(from_address, route.start_point)
        end_sim = calculate_similarity(to_address, route.end_point)
        
        rev_start_sim = calculate_similarity(from_address, route.end_point)
        rev_end_sim = calculate_similarity(to_address, route.start_point)

        direct_score = (start_sim * 0.5 + end_sim * 0.5) * 100
        return_score = (rev_start_sim * 0.5 + rev_end_sim * 0.5) * 100
        is_return = return_score > direct_score
        confidence = max(direct_score, return_score)

        is_exact = (
            norm_from == _normalize_location(route.start_point) and
            norm_to == _normalize_location(route.end_point)
        )

        fuzzy_recs.append({
            'id': route.id,
            'route_name': route.route_name,
            'start_point': route.start_point,
            'end_point': route.end_point,
            'fare': float(route.fare),
            'confidence': 100.0 if is_exact else round(confidence, 1),
            'is_exact': is_exact,
            'is_return_direction': is_return,
            'model_source': 'FuzzyStopMatcher'
        })

    # Combine & Sort: Exact matches first, then highest confidence descending, then fare ascending
    combined = ml_recs + fuzzy_recs
    combined.sort(key=lambda r: (-int(r['is_exact']), -r['confidence'], r['fare']))
    return combined[:top_n]


def find_best_route(from_address, to_address, college=None):
    """
    Finds the single best route for pass booking.
    Returns (route_object, confidence_score) or (None, 0.0).
    """
    from stc.models import BusRoute

    # 1. Exact DB match
    exact = BusRoute.objects.filter(
        start_point__iexact=from_address.strip(),
        end_point__iexact=to_address.strip()
    ).first()
    if exact:
        return exact, 100.0

    # 2. ML / Similarity Match
    recs = recommend_routes(from_address, to_address, college=college, top_n=1)
    if recs and recs[0]['confidence'] >= 45.0:
        route_obj = BusRoute.objects.filter(id=recs[0]['id']).first()
        return route_obj, recs[0]['confidence']

    return None, 0.0
