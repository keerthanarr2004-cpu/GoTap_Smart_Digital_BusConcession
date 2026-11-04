"""
Update stc/views.py with full ML integration:
- Imports: classify_issue_category, answer_student_question
- Feedback & Complaint saving with issue_category
- student_assistant view and api_student_assistant API
"""

import os

VIEWS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "stc", "views.py")

with open(VIEWS_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
old_imp = """from .ml_services import (
    recommend_routes,
    find_best_route,
    analyze_sentiment,
    get_sentiment_analytics,
    BusDemandPredictor,
)"""

new_imp = """from .ml_services import (
    recommend_routes,
    find_best_route,
    analyze_sentiment,
    classify_issue_category,
    get_sentiment_analytics,
    BusDemandPredictor,
    answer_student_question,
)"""

parts = content.split("# new")
if len(parts) > 1:
    parts[1] = parts[1].replace(old_imp, new_imp, 1)

    # 2. Update feedback view
    old_fb = """        analysis = analyze_sentiment(feedback_text)
        feedback_instance = Feedback(
            feedback_text=feedback_text,
            rating=rating,
            email=email,
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['score']
        )"""

    new_fb = """        analysis = analyze_sentiment(feedback_text)
        feedback_instance = Feedback(
            feedback_text=feedback_text,
            rating=rating,
            email=email,
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['score'],
            issue_category=analysis.get('issue_category', 'General Feedback')
        )"""

    parts[1] = parts[1].replace(old_fb, new_fb, 1)

    # 3. Update complaint view
    old_cp = """        analysis = analyze_sentiment(complaint_text)
        cp = models.Complaints(
            sname=sname, 
            semail=semail, 
            bname=bname, 
            complaint=complaint_text,
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['score']
        )"""

    new_cp = """        analysis = analyze_sentiment(complaint_text)
        cp = models.Complaints(
            sname=sname, 
            semail=semail, 
            bname=bname, 
            complaint=complaint_text,
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['score'],
            issue_category=analysis.get('issue_category', 'General Grievance')
        )"""

    parts[1] = parts[1].replace(old_cp, new_cp, 1)

    # 4. Check if student_assistant already present
    if "def student_assistant" not in parts[1]:
        assistant_views = """

# ============================================================================
# GoTap Student AI Assistant Views & APIs
# ============================================================================

def student_assistant(request):
    \"\"\"
    Interactive conversational AI assistant view for students.
    Provides guidance on passes, QR codes, routes, schedules, and payments.
    \"\"\"
    user_obj = None
    email = request.session.get('email')
    if email:
        user_obj = models.user.objects.filter(email=email).first()

    return render(request, 'student_assistant.html', {
        'user': user_obj
    })


@csrf_exempt
def api_student_assistant(request):
    \"\"\"
    REST API endpoint for GoTap Student AI Assistant chatbot.
    \"\"\"
    query = ''
    if request.method == 'POST':
        import json
        try:
            body = json.loads(request.body.decode('utf-8'))
            query = body.get('query', '')
        except Exception:
            query = request.POST.get('query', '')
    else:
        query = request.GET.get('query', '')

    user_obj = None
    email = request.session.get('email')
    if email:
        user_obj = models.user.objects.filter(email=email).first()

    response_data = answer_student_question(query, student_user=user_obj)
    return JsonResponse(response_data)
"""
        parts[1] = parts[1] + assistant_views

    new_content = "# new".join(parts)
    with open(VIEWS_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("stc/views.py successfully updated with AI/ML enhancements!")
else:
    print("Could not find '# new' anchor in stc/views.py")
