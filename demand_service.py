"""
Bus Demand Prediction Service for GoTap
======================================
Loads the Joblib-persisted Random Forest Regressor model and generates
real-time student passenger demand forecasts, weekly schedules,
rush hour indicators, and fleet dispatch recommendations.
"""

import os
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "demand_predictor.joblib")

# Global cached artifact
_DEMAND_BUNDLE = None


def get_demand_model():
    global _DEMAND_BUNDLE
    if _DEMAND_BUNDLE is None and os.path.exists(MODEL_PATH):
        try:
            _DEMAND_BUNDLE = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading demand predictor model: {e}")
            _DEMAND_BUNDLE = None
    return _DEMAND_BUNDLE


class BusDemandPredictor:
    """Bus Demand Predictor utilizing the trained Joblib RandomForestRegressor."""

    @classmethod
    def predict(cls, route_id, day_of_week=None, hour=None):
        from stc.models import BusRoute

        now = datetime.now()
        if day_of_week is None:
            day_of_week = now.weekday()
        if hour is None:
            hour = now.hour

        eval_hour = max(6, min(hour, 20))
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak = 1 if (7 <= eval_hour <= 9 or 16 <= eval_hour <= 18) else 0
        is_morning = 1 if 7 <= eval_hour <= 9 else 0
        is_evening = 1 if 16 <= eval_hour <= 18 else 0

        bundle = get_demand_model()
        pred_demand = 35

        if bundle and bundle.get('pipeline') is not None:
            try:
                pipeline = bundle['pipeline']
                features = pd.DataFrame([{
                    'route_id': route_id,
                    'day_of_week': day_of_week,
                    'hour': eval_hour,
                    'is_weekend': is_weekend,
                    'is_peak_hour': is_peak,
                    'is_monsoon': 0,
                    'active_pass_holders': 45,
                    'is_morning_peak': is_morning,
                    'is_evening_peak': is_evening
                }])
                pred_val = pipeline.predict(features)[0]
                pred_demand = int(round(max(2, min(pred_val, 85))))
            except Exception as e:
                print(f"Demand prediction inference fallback: {e}")
                pred_demand = 45 if is_peak else 20
        else:
            pred_demand = 45 if is_peak else 20

        # Categorize demand level
        if pred_demand >= 55:
            level = 'High'
            badge = 'danger'
            icon = '🔥'
            recommendation = 'High passenger rush expected. Deploy additional concession buses or allocate larger fleet.'
        elif pred_demand >= 30:
            level = 'Moderate'
            badge = 'warning'
            icon = '⚡'
            recommendation = 'Normal student concession rush. Standard bus schedule is optimal.'
        else:
            level = 'Low'
            badge = 'success'
            icon = '🟢'
            recommendation = 'Low passenger demand. Current fleet schedule is sufficient.'

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = days[day_of_week]
        capacity_pct = min(100, int((pred_demand / 55) * 100))

        return {
            'route_id': route_id,
            'day_of_week': day_of_week,
            'day_name': day_name,
            'hour': eval_hour,
            'hour_formatted': f"{eval_hour % 12 or 12}:00 {'AM' if eval_hour < 12 else 'PM'}",
            'predicted_passengers': pred_demand,
            'demand_level': level,
            'badge': badge,
            'icon': icon,
            'capacity_pct': capacity_pct,
            'recommendation': recommendation,
            'model_source': 'RandomForestRegressor'
        }

    @classmethod
    def get_weekly_forecast(cls, route_id):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        forecast = []
        for d in range(7):
            morning = cls.predict(route_id, day_of_week=d, hour=8)
            evening = cls.predict(route_id, day_of_week=d, hour=17)
            avg_demand = int((morning['predicted_passengers'] + evening['predicted_passengers']) / 2)
            level = 'High' if avg_demand >= 55 else ('Moderate' if avg_demand >= 30 else 'Low')
            badge = 'danger' if level == 'High' else ('warning' if level == 'Moderate' else 'success')
            
            forecast.append({
                'day_name': days[d],
                'avg_demand': avg_demand,
                'morning_peak': morning['predicted_passengers'],
                'evening_peak': evening['predicted_passengers'],
                'level': level,
                'badge': badge
            })
        return forecast

    @classmethod
    def get_all_routes_demand(cls, day_of_week=None, hour=None):
        from stc.models import BusRoute
        routes = list(BusRoute.objects.all())
        results = []
        for r in routes:
            pred = cls.predict(r.id, day_of_week=day_of_week, hour=hour)
            results.append({
                'route': r,
                'prediction': pred
            })
        results.sort(key=lambda x: -x['prediction']['predicted_passengers'])
        return results
