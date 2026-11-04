"""
NLP Sentiment Analysis & Issue Categorization Service for GoTap
==============================================================
Combines:
1. NLTK VADER Sentiment Intensity Analyzer for polarity and sentiment classification.
2. Scikit-learn TF-IDF + LogisticRegression Issue Classifier (Joblib) for 7 transit categories.
"""

import os
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "issue_classifier.joblib")

# Initialize VADER
try:
    _sia = SentimentIntensityAnalyzer()
except Exception:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    _sia = SentimentIntensityAnalyzer()

# Global cached issue classifier artifact
_ISSUE_BUNDLE = None


def get_issue_classifier():
    global _ISSUE_BUNDLE
    if _ISSUE_BUNDLE is None and os.path.exists(MODEL_PATH):
        try:
            _ISSUE_BUNDLE = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading issue classifier model: {e}")
            _ISSUE_BUNDLE = None
    return _ISSUE_BUNDLE


ISSUE_BADGES = {
    "Bus Delay / Punctuality": {"badge": "warning", "icon": "fas fa-clock"},
    "Overcrowding & Rush": {"badge": "danger", "icon": "fas fa-users"},
    "Conductor & Staff Behavior": {"badge": "danger", "icon": "fas fa-user-times"},
    "Pass, Payment & QR Scan Issue": {"badge": "info", "icon": "fas fa-qrcode"},
    "Cleanliness & Maintenance": {"badge": "secondary", "icon": "fas fa-broom"},
    "Route & Stop Coverage": {"badge": "primary", "icon": "fas fa-map-signs"},
    "Appreciation / Service Commendation": {"badge": "success", "icon": "fas fa-thumbs-up"},
    "General Feedback": {"badge": "secondary", "icon": "fas fa-comment"},
    "General Grievance": {"badge": "warning", "icon": "fas fa-exclamation-triangle"},
}


def classify_issue_category(text):
    """Classifies feedback or complaint text into an operational issue category."""
    if not text or not str(text).strip():
        return "General Feedback"

    bundle = get_issue_classifier()
    if bundle and bundle.get('pipeline') is not None:
        try:
            pred = bundle['pipeline'].predict([str(text)])[0]
            return pred
        except Exception as e:
            print(f"Issue classification error: {e}")

    # Heuristic fallback if model not loaded
    text_lower = str(text).lower()
    if any(k in text_lower for k in ['late', 'delay', 'time', 'timing', 'schedule', 'wait']):
        return "Bus Delay / Punctuality"
    if any(k in text_lower for k in ['crowd', 'rush', 'footboard', 'pack', 'seat', 'stand']):
        return "Overcrowding & Rush"
    if any(k in text_lower for k in ['conductor', 'driver', 'rude', 'shout', 'refuse', 'behave', 'misbehave']):
        return "Conductor & Staff Behavior"
    if any(k in text_lower for k in ['qr', 'code', 'wallet', 'payment', 'recharge', 'razorpay', 'pass']):
        return "Pass, Payment & QR Scan Issue"
    if any(k in text_lower for k in ['clean', 'dirty', 'leak', 'rain', 'broken', 'maintenance', 'window']):
        return "Cleanliness & Maintenance"
    if any(k in text_lower for k in ['stop', 'route', 'skip', 'missed', 'turn', 'direct']):
        return "Route & Stop Coverage"
    if any(k in text_lower for k in ['good', 'great', 'excellent', 'helpful', 'smooth', 'appreciate', 'fast', 'love']):
        return "Appreciation / Service Commendation"
    return "General Feedback"


def analyze_sentiment(text):
    """
    Analyzes sentiment and classifies operational transit issue category.
    Returns sentiment, polarity score, badges, and issue category.
    """
    if not text or not str(text).strip():
        return {
            'sentiment': 'Neutral',
            'score': 0.0,
            'badge': 'secondary',
            'emoji': '😐',
            'is_positive': False,
            'is_negative': False,
            'is_neutral': True,
            'issue_category': 'General Feedback',
            'issue_meta': ISSUE_BADGES.get('General Feedback')
        }

    scores = _sia.polarity_scores(str(text))
    compound = scores['compound']

    category = classify_issue_category(text)
    cat_meta = ISSUE_BADGES.get(category, {"badge": "secondary", "icon": "fas fa-tag"})

    # Determine sentiment: respect compound score, and refine domain context for transit grievances
    if compound >= 0.05 or (compound >= 0.0 and category == 'Appreciation / Service Commendation'):
        sentiment = 'Positive'
        badge = 'success'
        emoji = '😊'
    elif compound <= -0.05 or (compound == 0.0 and category in ['Bus Delay / Punctuality', 'Overcrowding & Rush', 'Conductor & Staff Behavior']):
        sentiment = 'Negative'
        badge = 'danger'
        emoji = '😟'
    else:
        sentiment = 'Neutral'
        badge = 'secondary'
        emoji = '😐'

    return {
        'sentiment': sentiment,
        'score': round(compound, 2),
        'badge': badge,
        'emoji': emoji,
        'is_positive': sentiment == 'Positive',
        'is_negative': sentiment == 'Negative',
        'is_neutral': sentiment == 'Neutral',
        'raw_scores': scores,
        'issue_category': category,
        'issue_meta': cat_meta
    }


def get_sentiment_analytics():
    """
    Calculates aggregated sentiment and issue categorization analytics
    across all stored student feedback and complaints in the database.
    """
    from stc.models import Feedback, Complaints

    feedbacks = list(Feedback.objects.all().order_by('-created_at'))
    complaints = list(Complaints.objects.all().order_by('-created'))

    fb_list = []
    comp_list = []

    pos_count = 0
    neg_count = 0
    neu_count = 0
    category_counts = {}

    for fb in feedbacks:
        text = fb.feedback_text
        analysis = analyze_sentiment(text)
        sent = fb.sentiment if (hasattr(fb, 'sentiment') and fb.sentiment) else analysis['sentiment']
        score = getattr(fb, 'sentiment_score', analysis['score'])
        cat = getattr(fb, 'issue_category', None) or analysis['issue_category']
        
        if sent == 'Positive':
            pos_count += 1
        elif sent == 'Negative':
            neg_count += 1
        else:
            neu_count += 1

        category_counts[cat] = category_counts.get(cat, 0) + 1

        fb_list.append({
            'obj': fb,
            'feedback_text': text,
            'rating': fb.rating,
            'email': fb.email,
            'created_at': fb.created_at,
            'sentiment': sent,
            'score': score,
            'badge': 'success' if sent == 'Positive' else ('danger' if sent == 'Negative' else 'secondary'),
            'emoji': '😊' if sent == 'Positive' else ('😟' if sent == 'Negative' else '😐'),
            'issue_category': cat,
            'issue_meta': ISSUE_BADGES.get(cat, {"badge": "secondary", "icon": "fas fa-tag"}),
        })

    for cp in complaints:
        text = cp.complaint
        analysis = analyze_sentiment(text)
        sent = cp.sentiment if (hasattr(cp, 'sentiment') and cp.sentiment) else analysis['sentiment']
        score = getattr(cp, 'sentiment_score', analysis['score'])
        cat = getattr(cp, 'issue_category', None) or analysis['issue_category']

        if sent == 'Negative':
            neg_count += 1
        elif sent == 'Positive':
            pos_count += 1
        else:
            neu_count += 1

        category_counts[cat] = category_counts.get(cat, 0) + 1

        comp_list.append({
            'obj': cp,
            'sname': cp.sname,
            'semail': cp.semail,
            'bname': cp.bname,
            'complaint': text,
            'created': cp.created,
            'sentiment': sent,
            'score': score,
            'badge': 'success' if sent == 'Positive' else ('danger' if sent == 'Negative' else 'secondary'),
            'emoji': '😊' if sent == 'Positive' else ('😟' if sent == 'Negative' else '😐'),
            'issue_category': cat,
            'issue_meta': ISSUE_BADGES.get(cat, {"badge": "secondary", "icon": "fas fa-tag"}),
        })

    total = len(fb_list) + len(comp_list)
    pos_pct = round((pos_count / total * 100), 1) if total > 0 else 0
    neg_pct = round((neg_count / total * 100), 1) if total > 0 else 0
    neu_pct = round((neu_count / total * 100), 1) if total > 0 else 0

    return {
        'total_feedback_and_complaints': total,
        'feedback_count': len(fb_list),
        'complaint_count': len(comp_list),
        'positive_count': pos_count,
        'negative_count': neg_count,
        'neutral_count': neu_count,
        'positive_pct': pos_pct,
        'negative_pct': neg_pct,
        'neutral_pct': neu_pct,
        'category_counts': category_counts,
        'feedbacks_with_sentiment': fb_list,
        'complaints_with_sentiment': comp_list,
    }
