"""
Student AI Assistant Service for GoTap
======================================
Provides conversational AI assistance to answer common student questions about:
1. Bus pass applications & Kerala student concession rules
2. QR-code verification, daily scan limits, and conductor terminals
3. Wallet recharges, Razorpay payments, and refund policies
4. Smart route recommendations & bus stop lookups
5. Filing complaints, tracking grievances, and feedback submission
6. Real-time personalized account inquiries (active pass & wallet balance)
"""

import re
from .route_service import recommend_routes, find_best_route


FAQ_KNOWLEDGE_BASE = [
    {
        "intent": "bus_pass_apply",
        "keywords": ["apply", "get pass", "concession", "eligibility", "documents", "validity", "how to book", "new pass", "renew"],
        "answer": (
            "### 🎫 How to Apply for a GoTap Student Bus Pass\n\n"
            "1. **Eligibility:** Regular enrolled students at recognized schools & colleges in Kerala.\n"
            "2. **How to Apply:**\n"
            "   - Log in to your student account.\n"
            "   - Go to **[Book Bus Pass](/book_bus/)**.\n"
            "   - Enter your Pickup stop (From) and Drop stop (To), or click one of our **AI Recommended Routes**.\n"
            "   - Confirm the route and pay the subsidized concession fare using your **GoTap Wallet**.\n"
            "3. **Instant Pass:** Once booked, your digital QR-coded bus pass is instantly generated in your account!"
        ),
        "actions": [
            {"label": "Book Bus Pass Now", "url": "/book_bus/", "icon": "fas fa-ticket-alt"},
            {"label": "Select Locations", "url": "/locations/", "icon": "fas fa-map-marker-alt"}
        ],
        "follow_ups": ["What documents are needed for student concession?", "How does the QR code bus pass work?", "How do I add money to my wallet?"]
    },
    {
        "intent": "qr_code_usage",
        "keywords": ["qr", "qr code", "scan", "scan qr", "conductor scan", "verify qr", "verify pass", "offline pass", "usage count", "how to use pass"],
        "answer": (
            "### 📱 How QR-Code Bus Pass Verification Works\n\n"
            "1. **On the Bus:** Open **[Book Bus Pass](/book_bus/)** or **[My Pass](/profile/)** to display your active QR code.\n"
            "2. **Conductor Scan:** Show the QR code to the conductor. The conductor scans it with the GoTap Conductor app.\n"
            "3. **Daily Limit:** Student concession passes allow **2 trips per day** (journey to college and return journey).\n"
            "4. **Offline Tip:** Take a screenshot of your active QR code on your phone so you can scan it even if mobile network is weak!"
        ),
        "actions": [
            {"label": "View My Active QR Pass", "url": "/book_bus/", "icon": "fas fa-qrcode"},
            {"label": "Student Profile", "url": "/profile/", "icon": "fas fa-user"}
        ],
        "follow_ups": ["What happens if QR code fails to scan?", "Can I use the pass more than twice a day?", "How do I report a conductor issue?"]
    },
    {
        "intent": "wallet_and_payment",
        "keywords": ["wallet", "balance", "recharge", "pay", "payment", "razorpay", "failed", "deducted", "money", "refund", "upi"],
        "answer": (
            "### 💳 GoTap Digital Wallet & Payments\n\n"
            "1. **Recharge Wallet:** Visit **[Show My Wallet](/showmywallet/)**, enter the amount, and pay securely using UPI, cards, or NetBanking powered by **Razorpay**.\n"
            "2. **Concession Fares:** Bus fares are heavily subsidized for students (₹2 to ₹10 for typical college corridors in Thrissur).\n"
            "3. **Payment Deduction Issue:** If money was debited from your bank account but wallet did not update within 5 minutes, please note your Razorpay payment ID and submit a complaint or contact RTO helpdesk."
        ),
        "actions": [
            {"label": "Open My Wallet", "url": "/showmywallet/", "icon": "fas fa-wallet"},
            {"label": "Transaction History", "url": "/user_transactions/", "icon": "fas fa-receipt"}
        ],
        "follow_ups": ["How much is the bus fare for students?", "How do I book a pass once wallet is recharged?"]
    },
    {
        "intent": "complaint_and_feedback",
        "keywords": ["complaint", "feedback", "conductor rude", "report", "delay", "late", "grievance", "rate", "driver", "footboard"],
        "answer": (
            "### ✍️ Filing Complaints & Feedback\n\n"
            "- **Submit Complaint:** If you encountered rude conductor behavior, bus skipping college stops, or overcrowding, submit details at **[File Complaint](/complaint/)**.\n"
            "- **Submit Feedback:** Share your rating and travel experience at **[Rate & Feedback](/feedback_rate/)**.\n"
            "- **AI Sentiment & Issue Tracking:** Our NLP engine automatically categorizes your feedback and flags urgent safety or conduct grievances directly on the RTO and Admin monitoring dashboards."
        ),
        "actions": [
            {"label": "File a Complaint", "url": "/complaint/", "icon": "fas fa-exclamation-circle"},
            {"label": "Submit Feedback", "url": "/feedback_rate/", "icon": "fas fa-star"}
        ],
        "follow_ups": ["Who investigates student bus complaints?", "How do I check my complaint status?"]
    },
    {
        "intent": "bus_timings_demand",
        "keywords": ["timing", "time", "schedule", "peak", "rush", "crowd", "morning", "evening", "hours", "busy"],
        "answer": (
            "### ⏰ Bus Timings & Peak Rush Hours\n\n"
            "- **Morning College Peak:** **7:30 AM – 9:30 AM** (High student rush, buses run every 10–15 mins).\n"
            "- **Afternoon Normal:** **12:00 PM – 2:30 PM** (Moderate passenger volume).\n"
            "- **Evening Return Peak:** **4:00 PM – 6:30 PM** (High rush returning from Thrissur colleges).\n"
            "- **AI Demand Forecast:** RTO uses our Machine Learning demand predictor to deploy extra concession buses during peak hours."
        ),
        "actions": [
            {"label": "View Routes", "url": "/locations/", "icon": "fas fa-route"}
        ],
        "follow_ups": ["Which routes connect to Kerala Varma College?", "How do I find the best bus route for my journey?"]
    }
]


def answer_student_question(user_query, student_user=None):
    """
    Answers student question using intent classification, keyword matching,
    route knowledge, and personal account status.
    """
    if not user_query or not str(user_query).strip():
        return {
            "reply": "Hello! I am your **GoTap AI Assistant**. How can I help you with your student bus pass, routes, QR verification, or payments today?",
            "intent": "greeting",
            "confidence": 100,
            "actions": [
                {"label": "Book Bus Pass", "url": "/book_bus/", "icon": "fas fa-ticket-alt"},
                {"label": "Check Wallet", "url": "/showmywallet/", "icon": "fas fa-wallet"},
                {"label": "Find Routes", "url": "/locations/", "icon": "fas fa-route"}
            ],
            "follow_ups": [
                "How do I apply for a bus pass?",
                "How do QR codes work on the bus?",
                "What routes go to Kerala Varma College?",
                "How do I recharge my wallet?"
            ]
        }

    q = str(user_query).lower().strip()

    # 1. Greetings
    if any(q.startswith(g) for g in ['hi', 'hello', 'hey', 'good morning', 'good evening', 'namaskaram']):
        name_str = f", {student_user.name}" if (student_user and getattr(student_user, 'name', None)) else ""
        return {
            "reply": f"Hello{name_str}! 👋 Welcome to **GoTap AI Assistant**. I can help you with:\n- Booking and renewing bus passes\n- Conductor QR code verification\n- Checking fares and recommended routes\n- Wallet recharges and payment issues\n- Submitting complaints & feedback\n\nWhat would you like to know?",
            "intent": "greeting",
            "confidence": 98,
            "actions": [
                {"label": "Book Pass", "url": "/book_bus/", "icon": "fas fa-ticket-alt"},
                {"label": "My Wallet", "url": "/showmywallet/", "icon": "fas fa-wallet"}
            ],
            "follow_ups": ["How do I book a pass?", "Show me routes to my college", "How to recharge wallet?"]
        }

    # 2. Account Status Query (e.g. "my pass", "my wallet balance", "check status")
    if student_user and any(k in q for k in ['my pass', 'my wallet', 'my balance', 'my account', 'active pass', 'have a pass']):
        from stc.models import BusPass, Wallet
        active_pass = BusPass.objects.filter(user=student_user, is_paid=True).first()
        wallets = Wallet.objects.filter(wuser=student_user)
        total_balance = sum(w.amount for w in wallets) if wallets.exists() else 0.0

        pass_info = f"✅ You have an **Active Pass** from **{active_pass.from_address}** to **{active_pass.to_address}** (Amount: ₹{active_pass.amount})." if active_pass else "ℹ️ You currently do **not have an active bus pass**."
        
        reply = (
            f"### 👤 Account Overview for {student_user.name or student_user.email}\n\n"
            f"- **Wallet Balance:** ₹{total_balance:.2f}\n"
            f"- **Bus Pass Status:** {pass_info}\n"
            f"- **Registered College:** {student_user.college or 'Not specified'}\n"
            f"- **Home Route:** {student_user.from_address or 'Not set'} ➔ {student_user.to_address or 'Not set'}"
        )
        return {
            "reply": reply,
            "intent": "account_status",
            "confidence": 95,
            "actions": [
                {"label": "View QR Pass", "url": "/book_bus/", "icon": "fas fa-qrcode"} if active_pass else {"label": "Book Pass Now", "url": "/book_bus/", "icon": "fas fa-ticket-alt"},
                {"label": "Recharge Wallet", "url": "/showmywallet/", "icon": "fas fa-wallet"}
            ],
            "follow_ups": ["How to book a new pass?", "How to recharge my wallet?"]
        }

    # 3. Route Recommendation Query (e.g., "how to reach ollur", "route to kerala varma", "bus to chalakudy")
    route_match = re.search(r'(?:to|towards|for|reach|go to)\s+([a-zA-Z\s]+)', q)
    if any(k in q for k in ['route', 'bus to', 'reach', 'how to go', 'travel to', 'stops']) and route_match:
        dest_candidate = route_match.group(1).strip()
        from_stop = student_user.from_address if (student_user and getattr(student_user, 'from_address', None)) else "Kerala Varma College"
        recs = recommend_routes(from_stop, dest_candidate, top_n=3)
        if recs:
            lines = [f"- **{r['start_point']} ➔ {r['end_point']}** ({r['route_name']}) | Fare: **₹{r['fare']}** | Match: **{r['confidence']}%**" for r in recs]
            reply = (
                f"### 🚌 Recommended Bus Routes to '{dest_candidate.title()}'\n\n"
                + "\n".join(lines) +
                f"\n\nYou can book any of these routes directly on GoTap with student concession!"
            )
            return {
                "reply": reply,
                "intent": "route_inquiry",
                "confidence": 90,
                "actions": [
                    {"label": f"Book Pass to {dest_candidate.title()}", "url": f"/book_bus/?to={dest_candidate}", "icon": "fas fa-ticket-alt"}
                ],
                "follow_ups": ["How much is the student bus pass fare?", "How does the QR verification work?"]
            }

    # 4. Search Knowledge Base
    best_match = None
    max_score = 0

    for item in FAQ_KNOWLEDGE_BASE:
        score = 0
        for kw in item["keywords"]:
            if kw in q:
                score += len(kw.split()) * 2
        if score > max_score:
            max_score = score
            best_match = item

    if best_match and max_score >= 2:
        return {
            "reply": best_match["answer"],
            "intent": best_match["intent"],
            "confidence": min(95, 60 + max_score * 5),
            "actions": best_match.get("actions", []),
            "follow_ups": best_match.get("follow_ups", [])
        }

    # 5. Smart Fallback
    fallback_reply = (
        "I'm here to help with all questions about **GoTap Smart Transportation**!\n\n"
        "Here are common things I can help you with:\n"
        "- 🎫 **Bus Pass Booking & Renewal:** How to select routes and apply for concession passes.\n"
        "- 📱 **QR Code Verification:** How to scan and validate your pass on the conductor terminal.\n"
        "- 💳 **Wallet & Payments:** How to recharge your wallet balance via Razorpay UPI/Netbanking.\n"
        "- 🚌 **Route Finder:** Look up buses connecting Thrissur, Ollur, Irinjalakuda, Chalakudy, and colleges.\n"
        "- 📝 **Complaints & Feedback:** How to lodge a grievance about conductor behavior or delays."
    )
    return {
        "reply": fallback_reply,
        "intent": "general_help",
        "confidence": 50,
        "actions": [
            {"label": "Book Bus Pass", "url": "/book_bus/", "icon": "fas fa-ticket-alt"},
            {"label": "Recharge Wallet", "url": "/showmywallet/", "icon": "fas fa-wallet"},
            {"label": "File Complaint", "url": "/complaint/", "icon": "fas fa-exclamation-circle"}
        ],
        "follow_ups": [
            "How do I apply for a bus pass?",
            "How does QR code verification work?",
            "How do I recharge my wallet?",
            "Which routes connect to Kerala Varma College?"
        ]
    }
