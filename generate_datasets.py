"""
Dataset Generator for GoTap Smart Student Transportation System
==============================================================
Generates realistic, high-quality domain datasets for:
1. Route Recommendation (Student travel features -> Recommended Route ID)
2. Bus Demand Prediction (Route, temporal & transit factors -> Hourly Passenger Demand)
3. NLP Feedback and Complaint Analysis (Student reviews -> Sentiment & Issue Category)
"""

import os
import random
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Actual Bus Routes from GoTap Database
DB_ROUTES = [
    {"id": 158, "name": "city route", "start": "Kerala Varma College", "end": "Ollur", "fare": 5.0, "dist": 6.8},
    {"id": 159, "name": "round route", "start": "Kerala Varma College", "end": "thanikudam", "fare": 5.0, "dist": 10.2},
    {"id": 160, "name": "mala road", "start": "Kerala Varma College", "end": "thrissur", "fare": 5.0, "dist": 4.5},
    {"id": 161, "name": "kavil road", "start": "Kerala Varma College", "end": "irinjakkuda", "fare": 5.0, "dist": 21.0},
    {"id": 162, "name": "junction", "start": "Kerala Varma College", "end": "maala", "fare": 5.0, "dist": 29.5},
    {"id": 163, "name": "city route", "start": "Kerala Varma College", "end": "manuthy", "fare": 5.0, "dist": 8.0},
    {"id": 164, "name": "kizhak", "start": "Kerala Varma College", "end": "ayanthole", "fare": 5.0, "dist": 3.8},
    {"id": 165, "name": "junction", "start": "Kerala Varma College", "end": "chalakkudy", "fare": 5.0, "dist": 32.0},
    {"id": 166, "name": "city route", "start": "Kerala Varma College", "end": "kodakara", "fare": 10.0, "dist": 19.5},
    {"id": 167, "name": "junction", "start": "Kerala Varma College", "end": "viyoor", "fare": 10.0, "dist": 5.2},
    {"id": 168, "name": "city route", "start": "Kerala Varma College", "end": "kunnamkulam", "fare": 2.0, "dist": 24.0},
    {"id": 169, "name": "junction", "start": "Kerala Varma College", "end": "kechery", "fare": 2.0, "dist": 16.5},
    {"id": 170, "name": "city route", "start": "Kerala Varma College", "end": "Guruvayoor", "fare": 5.0, "dist": 28.0},
    {"id": 175, "name": "city route", "start": "Vimala college", "end": "irinjakkuda", "fare": 10.0, "dist": 22.5},
    {"id": 180, "name": "city route", "start": "Ollur", "end": "Kerala Varma College", "fare": 5.0, "dist": 6.8},
    {"id": 181, "name": "city route", "start": "manuthy", "end": "Kerala Varma College", "fare": 5.0, "dist": 8.0},
    {"id": 182, "name": "round route", "start": "thanikudam", "end": "Kerala Varma College", "fare": 5.0, "dist": 10.2},
    {"id": 183, "name": "mala road", "start": "thrissur", "end": "Kerala Varma College", "fare": 5.0, "dist": 4.5},
    {"id": 184, "name": "kavil road", "start": "irinjakkuda", "end": "Kerala Varma College", "fare": 5.0, "dist": 21.0},
    {"id": 185, "name": "junction", "start": "maala", "end": "Kerala Varma College", "fare": 5.0, "dist": 29.5},
    {"id": 186, "name": "kizhak", "start": "ayanthole", "end": "Kerala Varma College", "fare": 5.0, "dist": 3.8},
    {"id": 187, "name": "junction", "start": "chalakkudy", "end": "Kerala Varma College", "fare": 5.0, "dist": 32.0},
    {"id": 188, "name": "city route", "start": "kodakara", "end": "Kerala Varma College", "fare": 10.0, "dist": 19.5},
    {"id": 189, "name": "junction", "start": "viyoor", "end": "Kerala Varma College", "fare": 10.0, "dist": 5.2},
    {"id": 190, "name": "city route", "start": "kunnamkulam", "end": "Kerala Varma College", "fare": 2.0, "dist": 24.0},
    {"id": 191, "name": "junction", "start": "kechery", "end": "Kerala Varma College", "fare": 2.0, "dist": 16.5},
    {"id": 192, "name": "city route", "start": "Guruvayoor", "end": "Kerala Varma College", "fare": 5.0, "dist": 28.0},
    {"id": 193, "name": "city route", "start": "irinjakkuda", "end": "Vimala college", "fare": 10.0, "dist": 22.5},
]

COLLEGES = [
    "Sree Kerala Varma College",
    "Vimala College Thrissur",
    "St. Thomas College Thrissur",
    "Govt Engineering College Thrissur",
    "Sree Krishna College Guruvayur",
    "St. Aloysius College Elthuruth",
    "Christ College Irinjalakuda",
]

COURSES = ["BSc Computer Science", "BCom", "BCA", "BTech", "BA English", "MSc Mathematics", "BBA"]

STOP_VARIATIONS = {
    "Kerala Varma College": ["Kerala Varma College", "kerala varma", "KVC Thrissur", "Kanattukara KVC", "K.V. College"],
    "Vimala college": ["Vimala college", "vimala", "Vimala College Ramavarmapuram", "Vimala junction"],
    "Ollur": ["Ollur", "olur", "Ollur Junction", "Ollur church stop"],
    "thanikudam": ["thanikudam", "Thanikkudam", "Thanikudam bridge"],
    "thrissur": ["thrissur", "Trichur Round", "Thrissur Sakthan Stand", "Thrissur North Stand", "TCR Bus Stand"],
    "irinjakkuda": ["irinjakkuda", "Irinjalakuda", "ijk", "Irinjalakuda Bus Stand", "Christ College stop"],
    "maala": ["maala", "Mala", "Mala Bus Stand", "Mala town"],
    "manuthy": ["manuthy", "Mannuthy", "Mannuthy Highway JN", "Veterinary College stop"],
    "ayanthole": ["ayanthole", "Ayyanthole", "Civil Station Ayyanthole", "Ayyanthole Collectorate"],
    "chalakkudy": ["chalakkudy", "Chalakudy", "Chalakudy Bus Stand", "Chalakudy South"],
    "kodakara": ["kodakara", "Kodakara", "Kodakara Underpass", "Kodakara Flyover"],
    "viyoor": ["viyoor", "Viyoor", "Viyoor Bridge", "Viyoor Jail Jn"],
    "kunnamkulam": ["kunnamkulam", "Kunnamkulam", "Kunnamkulam Town", "Kunnamkulam Bus Stand"],
    "kechery": ["kechery", "Kechery", "Kecheri", "Kechery Centre"],
    "Guruvayoor": ["Guruvayoor", "Guruvayur", "Guruvayur West Nada", "Guruvayur Private Stand"],
}


def generate_route_travel_dataset(n_samples=2000):
    """Generate labeled route travel data for student route recommendation."""
    records = []
    
    for _ in range(n_samples):
        # Pick a target route
        target_route = random.choice(DB_ROUTES)
        
        start_variants = STOP_VARIATIONS.get(target_route["start"], [target_route["start"]])
        end_variants = STOP_VARIATIONS.get(target_route["end"], [target_route["end"]])
        
        origin_text = random.choice(start_variants)
        dest_text = random.choice(end_variants)
        
        # Add random travel attributes
        college = "Sree Kerala Varma College" if "Kerala Varma" in target_route["start"] or "Kerala Varma" in target_route["end"] else random.choice(COLLEGES)
        course = random.choice(COURSES)
        student_year = random.randint(1, 4)
        travel_hour = random.choice([7, 8, 8, 9, 9, 12, 13, 16, 16, 17, 17, 18])
        day_of_week = random.randint(0, 6)
        
        # Add slight noise to distance and fare
        noise_dist = max(1.0, round(target_route["dist"] + random.uniform(-0.8, 0.8), 2))
        
        records.append({
            "origin_stop": origin_text,
            "destination_stop": dest_text,
            "canonical_origin": target_route["start"],
            "canonical_destination": target_route["end"],
            "distance_km": noise_dist,
            "fare": target_route["fare"],
            "travel_hour": travel_hour,
            "day_of_week": day_of_week,
            "student_year": student_year,
            "college": college,
            "course": course,
            "target_route_id": target_route["id"],
            "target_route_name": target_route["name"],
        })
        
    df = pd.DataFrame(records)
    out_path = os.path.join(RAW_DATA_DIR, "route_travel_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} route travel records -> {out_path}")
    return df


def generate_bus_demand_dataset(n_days=90):
    """Generate historical hourly bus demand data across all routes."""
    records = []
    start_date = datetime(2026, 8, 1)  # Recent semester start
    
    for d in range(n_days):
        current_date = start_date + timedelta(days=d)
        dow = current_date.weekday()
        is_weekend = 1 if dow >= 5 else 0
        is_sunday = 1 if dow == 6 else 0
        
        # Monsoon season in Kerala (Aug-Sept has higher transit demand due to rain)
        is_monsoon = 1 if current_date.month in [8, 9] else 0
        
        for route in DB_ROUTES:
            # Hub popularity weight
            base_riders = 20.0
            if "thrissur" in route["start"].lower() or "thrissur" in route["end"].lower():
                base_riders += 12.0
            if "chalakkudy" in route["start"].lower() or "chalakkudy" in route["end"].lower():
                base_riders += 8.0
            if "irinjakkuda" in route["start"].lower() or "irinjakkuda" in route["end"].lower():
                base_riders += 10.0
                
            active_pass_holders = int(base_riders * random.uniform(1.8, 2.5))
            
            for hour in range(6, 21):
                # Peak hours: Morning college rush (7:30 to 9:30 AM), Evening rush (4:00 to 6:30 PM)
                if 7 <= hour <= 9:
                    peak_factor = 2.15
                    is_peak = 1
                elif 16 <= hour <= 18:
                    peak_factor = 1.95
                    is_peak = 1
                elif 12 <= hour <= 14:
                    peak_factor = 1.15
                    is_peak = 0
                else:
                    peak_factor = 0.42
                    is_peak = 0
                
                # Weekend reduction (colleges closed Saturday afternoon / Sunday)
                dow_factor = 0.25 if is_sunday else (0.60 if is_weekend else 1.0)
                weather_factor = 1.15 if is_monsoon else 1.0
                
                mean_demand = base_riders * peak_factor * dow_factor * weather_factor
                actual_demand = int(np.random.normal(mean_demand, 3.2))
                actual_demand = max(1, min(actual_demand, 85))  # Plausible bus capacity limit
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "route_id": route["id"],
                    "route_name": route["name"],
                    "start_point": route["start"],
                    "end_point": route["end"],
                    "day_of_week": dow,
                    "hour": hour,
                    "is_weekend": is_weekend,
                    "is_peak_hour": is_peak,
                    "is_monsoon": is_monsoon,
                    "active_pass_holders": active_pass_holders,
                    "passenger_demand": actual_demand,
                })
                
    df = pd.DataFrame(records)
    out_path = os.path.join(RAW_DATA_DIR, "historical_bus_demand.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} historical bus demand records -> {out_path}")
    return df


def generate_feedback_complaint_dataset():
    """Generate domain-rich NLP dataset of student transit feedback & complaints with sentiment and issue category."""
    samples = [
        # 1. Bus Delay / Punctuality
        ("The morning bus on Ollur to Kerala Varma route is consistently 25 minutes late, making us miss the first period lecture.", "Negative", 1, "Bus Delay / Punctuality"),
        ("Bus was delayed by more than half an hour today at Sakthan stand without any notification.", "Negative", 1, "Bus Delay / Punctuality"),
        ("Frequent delay during evening trips from Thrissur to Irinjalakuda. Very frustrating.", "Negative", 2, "Bus Delay / Punctuality"),
        ("The bus arrived 20 minutes behind schedule at Mannuthy stop today morning.", "Negative", 2, "Bus Delay / Punctuality"),
        ("Bus timing is erratic on the Chalakudy route. Please fix the schedule.", "Negative", 2, "Bus Delay / Punctuality"),
        ("Punctual departure today from Vimala college, arrived right on time.", "Positive", 5, "Bus Delay / Punctuality"),
        ("Bus arrived exactly on schedule at Kunnamkulam stop today, very impressed.", "Positive", 5, "Bus Delay / Punctuality"),
        ("Bus was slightly delayed by 5 minutes due to heavy traffic near Swaraj Round.", "Neutral", 3, "Bus Delay / Punctuality"),

        # 2. Overcrowding & Rush
        ("Extremely overcrowded during 8:30 AM trip to Kerala Varma. Students are standing on footboard which is very dangerous.", "Negative", 1, "Overcrowding & Rush"),
        ("Bus is completely packed like sardines between Sakthan and Ollur, conductors don't stop for girl students.", "Negative", 1, "Overcrowding & Rush"),
        ("Heavy rush during evening 5 PM bus to Irinjalakuda. We need an extra concession bus on this route.", "Negative", 2, "Overcrowding & Rush"),
        ("Could barely enter the bus at Ayyanthole due to extreme student crowding.", "Negative", 2, "Overcrowding & Rush"),
        ("Too many passengers standing in the aisle, very suffocating and risky travel.", "Negative", 1, "Overcrowding & Rush"),
        ("Adequate seating capacity available on the 10 AM trip, very comfortable.", "Positive", 4, "Overcrowding & Rush"),
        ("The extra bus deployed during exam week reduced the overcrowding significantly. Excellent initiative!", "Positive", 5, "Overcrowding & Rush"),
        ("Moderate crowd on the 2 PM trip, standing room was manageable.", "Neutral", 3, "Overcrowding & Rush"),

        # 3. Conductor & Staff Behavior
        ("Conductor was extremely rude when I showed my GoTap digital bus pass and demanded physical cash.", "Negative", 1, "Conductor & Staff Behavior"),
        ("Bus staff refused to acknowledge student concession pass and threatened to deboard me.", "Negative", 1, "Conductor & Staff Behavior"),
        ("Conductor misbehaved with first year students near Kodakara stop. Very bad attitude.", "Negative", 1, "Conductor & Staff Behavior"),
        ("Conductor refused to stop the bus properly at Kerala Varma gate, stopped 200m away in rain.", "Negative", 2, "Conductor & Staff Behavior"),
        ("Very polite and helpful conductor on the Guruvayur route, verified QR pass politely.", "Positive", 5, "Conductor & Staff Behavior"),
        ("The bus driver and conductor were very courteous and patient with students.", "Positive", 5, "Conductor & Staff Behavior"),
        ("Conductor checked the pass without any issue today.", "Neutral", 3, "Conductor & Staff Behavior"),

        # 4. Pass, Payment & QR Scan Issue
        ("My QR code failed to scan on conductor terminal multiple times today, had to pay full ticket.", "Negative", 1, "Pass, Payment & QR Scan Issue"),
        ("Money was deducted from my wallet via Razorpay but bus pass was not renewed.", "Negative", 1, "Pass, Payment & QR Scan Issue"),
        ("QR scan was showing invalid pass even though I paid the concession fee yesterday.", "Negative", 1, "Pass, Payment & QR Scan Issue"),
        ("Wallet balance deducted twice for single journey pass. Please refund immediately.", "Negative", 1, "Pass, Payment & QR Scan Issue"),
        ("Instant QR scan and seamless payment recharge! The GoTap digital wallet works flawlessly.", "Positive", 5, "Pass, Payment & QR Scan Issue"),
        ("Renewed my monthly concession pass in 30 seconds online. QR code verified immediately on bus.", "Positive", 5, "Pass, Payment & QR Scan Issue"),
        ("QR code scanning took about 10 seconds to respond, but it verified successfully.", "Neutral", 3, "Pass, Payment & QR Scan Issue"),

        # 5. Cleanliness & Bus Condition
        ("Bus interior is extremely dirty, seats are torn and window glass was rattling dangerously.", "Negative", 1, "Cleanliness & Bus Condition"),
        ("Rain water was leaking through the roof onto student seats in bus KL-08-3341.", "Negative", 1, "Cleanliness & Bus Condition"),
        ("Broken handrails inside the bus made it difficult to balance during sudden braking.", "Negative", 2, "Cleanliness & Bus Condition"),
        ("Very clean bus, functioning fans, and well-maintained seats on the Mala route.", "Positive", 5, "Cleanliness & Bus Condition"),
        ("Neat and hygienic bus interior. Appreciate the depot maintenance team.", "Positive", 5, "Cleanliness & Bus Condition"),
        ("Bus condition is average, cleanliness could be slightly improved.", "Neutral", 3, "Cleanliness & Bus Condition"),

        # 6. Route & Stop Coverage
        ("Bus bypassed the college bus stop without stopping today morning, leaving 15 students stranded.", "Negative", 1, "Route & Stop Coverage"),
        ("Need a direct concession route from Thanikkudam to Kerala Varma College without bus changes.", "Negative", 2, "Route & Stop Coverage"),
        ("The bus took a diversion and skipped the Viyoor stop without informing passengers.", "Negative", 2, "Route & Stop Coverage"),
        ("The route covers all major student stops conveniently. Great connectivity for hostelers.", "Positive", 5, "Route & Stop Coverage"),
        ("Good connectivity between Vimala College and Irinjalakuda bus terminus.", "Positive", 5, "Route & Stop Coverage"),
        ("Bus route follows the regular highway itinerary as expected.", "Neutral", 3, "Route & Stop Coverage"),

        # 7. Appreciation / Service Commendation
        ("GoTap digital concession system has made student bus travel so convenient and hassle-free!", "Positive", 5, "Appreciation / Service Commendation"),
        ("Fantastic service! No more waiting in long queues at KSRTC depot for paper passes.", "Positive", 5, "Appreciation / Service Commendation"),
        ("Smart route recommendation suggested the exact bus I needed for college. Love this app!", "Positive", 5, "Appreciation / Service Commendation"),
        ("Excellent initiative by college transportation department. Fast, smooth, and paperless.", "Positive", 5, "Appreciation / Service Commendation"),
        ("Good digital bus pass application for Kerala college students.", "Positive", 4, "Appreciation / Service Commendation"),
        ("Service is satisfactory and meets daily student commuting requirements.", "Neutral", 3, "Appreciation / Service Commendation"),
    ]

    # Expand dataset with variations to create a robust corpus of 600+ records
    expanded = []
    modifiers = [
        "", "Today ", "Yesterday ", "This week, ", "Usually ", "Every morning ", 
        "During rush hours, ", "On my daily commute, ", "Honestly, "
    ]
    suffixes = [
        "", " Please resolve this as soon as possible.", " Kindly take prompt action.",
        " Overall a good experience.", " Really appreciate the system.", " Needs prompt attention from RTO."
    ]

    for base_text, sent, rating, category in samples:
        for _ in range(12):
            mod = random.choice(modifiers)
            suf = random.choice(suffixes) if sent == "Negative" else ("!" if sent == "Positive" else ".")
            text = f"{mod}{base_text.lower() if mod else base_text}{suf}".strip()
            
            # Add occasional student metadata
            expanded.append({
                "text": text,
                "sentiment": sent,
                "rating": rating,
                "issue_category": category,
            })

    df = pd.DataFrame(expanded)
    out_path = os.path.join(RAW_DATA_DIR, "feedback_complaint_nlp.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} feedback/complaint NLP records -> {out_path}")
    return df


if __name__ == "__main__":
    print("Generating GoTap AI/ML Raw Datasets...")
    generate_route_travel_dataset(2200)
    generate_bus_demand_dataset(90)
    generate_feedback_complaint_dataset()
    print("All datasets generated successfully!")
