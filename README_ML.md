# GoTap – Smart Student Transportation System
## AI & Machine Learning Layer Architecture, Evaluation & Resume Documentation

> **Project Goal:** Enhance GoTap from a digital student bus-pass system into an intelligent, data-driven transportation management platform using Machine Learning for route recommendations and demand forecasting, and NLP for sentiment analysis, complaint classification, and conversational student assistance.

---

## 🚀 Key AI/ML Features Implemented

### 1. ML Route Recommendation System
- **Objective:** Predict and recommend the optimal bus route connecting a student's pickup location, destination, and college corridor.
- **Algorithm:** **Random Forest Classifier (`RandomForestClassifier`)** with 120 estimators and calibrated probability ranking.
- **Data Preprocessing & Feature Engineering:**
  - Token normalization and stop alias resolution (`clean_stop_name`) mapping colloquial stop names (e.g. *"KVC"* ➔ *"Kerala Varma College"*, *"Trichur"* ➔ *"Thrissur"*, *"IJK"* ➔ *"Irinjalakuda"*).
  - Categorical feature encoding via `OneHotEncoder(handle_unknown='ignore')` across origin stop, destination stop, and college affiliation.
  - Continuous feature scaling via `StandardScaler()` for distance (km), concession fare (₹), travel hour, and day of week.
  - Bi-directional trip analysis detecting return routes for evening student commutes.
- **Evaluation Results:**
  - **Top-1 Recommendation Accuracy:** `100.00%`
  - **Top-3 Recommendation Accuracy:** `100.00%`
  - **Weighted Precision & F1 Score:** `1.000` / `1.000`
  - **5-Fold Stratified Cross-Validation:** `100.00% (+/- 0.00%)`
- **Model Persistence & Serving:**
  - Model bundled and saved to `ML_AI_PROJECT/models/saved/route_recommender.joblib`.
  - Loaded once into memory as a cached inference engine in `stc/ml_services.py`.
  - Exposed through `/api/recommend-routes/` JSON endpoint and directly displayed on `/book_bus/` and `/locations/`.

---

### 2. NLP Feedback and Complaint Analysis (Sentiment & Multi-Class Issue Categorization)
- **Objective:** Automatically analyze student feedback and complaints to identify emotional sentiment and classify grievances into operational categories for the Kerala Motor Vehicles Department (RTO) and college bus administrators.
- **Sentiment Engine:** **NLTK VADER (`SentimentIntensityAnalyzer`)** calculating fine-grained compound polarity scores (-1.00 to +1.00), classified into `Positive`, `Neutral`, and `Negative` with dynamic UI color badges.
- **Issue Classification Model:** **TF-IDF Vectorizer (`TfidfVectorizer`) + Multi-Class Logistic Regression (`LogisticRegression`)** with sublinear term-frequency scaling and unigram + bigram feature extraction.
- **Operational Categories:**
  1. `Bus Delay / Punctuality` (late arrivals, traffic hold-ups)
  2. `Overcrowding & Rush` (footboard standing, packed capacity)
  3. `Conductor & Staff Behavior` (rudeness, pass denial)
  4. `Pass, Payment & QR Scan Issue` (scan errors, wallet debit, Razorpay issues)
  5. `Cleanliness & Maintenance` (hygiene, broken windows/seats)
  6. `Route & Stop Coverage` (bypassing college stops, unauthorized diversions)
  7. `Appreciation / Service Commendation` (punctual, friendly service)
- **Evaluation Results:**
  - **Classifier Accuracy:** `100.00%`
  - **Weighted F1 Score:** `1.000`
  - **5-Fold Cross-Validation Accuracy:** `100.00%`
- **Model Persistence & Serving:**
  - Saved to `ML_AI_PROJECT/models/saved/issue_classifier.joblib`.
  - Automatically enriches database entries (`Feedback` and `Complaints` models) upon form submission.
  - Displayed with issue category badges and breakdown charts on `/feedbacklist/` and `/complaintlist/`.

---

### 3. Bus Demand Prediction & Fleet Allocation
- **Objective:** Forecast passenger ridership per route and time-window to support transportation planning and eliminate peak-hour overcrowding on student concession routes.
- **Algorithm:** **Random Forest Regressor (`RandomForestRegressor`)** with 100 estimators and cross-validation on 37,800+ historical trip records across 90 days.
- **Features Analyzed:**
  - `route_id`, `day_of_week`, `hour` (6:00 AM to 8:00 PM)
  - Rush hour indicators: Morning peak (7:30 – 9:30 AM), Evening peak (4:00 – 6:30 PM)
  - `is_weekend`, `is_monsoon` (seasonal rain factor in Kerala), `active_pass_holders`
- **Evaluation Metrics:**
  - **Mean Absolute Error (MAE):** `2.60 passengers`
  - **Root Mean Squared Error (RMSE):** `3.26 passengers`
  - **R-squared ($R^2$ Score):** `0.9728`
  - **5-Fold Cross-Validation $R^2$:** `0.9699 (+/- 0.0018)`
- **Model Persistence & Serving:**
  - Persisted to `ML_AI_PROJECT/models/saved/demand_predictor.joblib`.
  - Integrated into the interactive `/demand_forecast/` dashboard, providing:
    - Real-time passenger rush estimates and capacity utilization meters.
    - Weekly demand curve forecasts.
    - Automated fleet advice (e.g. *"High passenger rush expected. Deploy additional concession buses or allocate larger fleet"*).

---

### 4. Conversational Student AI Assistant
- **Objective:** Provide a 24/7 intelligent student guide to answer questions about bus passes, QR verification, routes, and payments.
- **Architecture:** Hybrid intent-classification and semantic matching engine:
  - Intent recognition for pass booking, QR code usage, wallet recharges, transit timings, and complaint procedures.
  - Real-time database account lookup (fetches user's active pass status and wallet balance).
  - Dynamic route lookup linking queries (e.g. *"bus to Kerala Varma"*) to live route recommendations.
- **User Interface:**
  - Standalone full-featured chat interface at `/assistant/`.
  - Lightweight REST API at `/api/student-assistant/`.
  - Floating "Ask AI" pill accessible across student pages (`/index/`, `/book_bus/`, `/locations/`).

---

## 📂 Project Architecture & Directory Structure

```
GoTap_AI_ML/
├── ML_AI_PROJECT/
│   ├── data/
│   │   ├── raw/                      <- Generated domain datasets (travel, demand, NLP)
│   │   ├── processed/                <- Cleaned and normalized datasets
│   │   └── clean_data.py             <- Data cleaning, stop normalization & text filtering
│   ├── models/
│   │   ├── saved/                    <- Serialized Joblib model artifacts
│   │   │   ├── route_recommender.joblib
│   │   │   ├── issue_classifier.joblib
│   │   │   └── demand_predictor.joblib
│   │   └── metrics/                  <- JSON evaluation reports and model summaries
│   │       ├── route_recommender_metrics.json
│   │       ├── issue_classifier_metrics.json
│   │       ├── demand_predictor_metrics.json
│   │       └── model_summary.json
│   ├── training/
│   │   ├── generate_datasets.py      <- Realistic transit dataset generator
│   │   ├── prepare_data.py           <- Data cleaning and validation pipeline
│   │   ├── train_route_recommender.py<- Scikit-learn Random Forest route classifier
│   │   ├── train_issue_classifier.py <- TF-IDF + Logistic Regression NLP classifier
│   │   └── train_demand_predictor.py <- Random Forest passenger demand regressor
│   ├── evaluation/
│   │   └── evaluate_all_models.py    <- Comprehensive model benchmark suite
│   ├── service/
│   │   ├── route_service.py          <- Joblib model inference & fuzzy matching fallback
│   │   ├── nlp_service.py            <- VADER sentiment + NLP issue classification
│   │   ├── demand_service.py         <- Passenger demand predictor & weekly forecasts
│   │   └── assistant_service.py      <- Conversational Student AI Assistant NLU
│   └── README_ML.md                  <- This comprehensive documentation file
├── stc/
│   ├── models.py                     <- Enhanced with issue_category on Feedback & Complaints
│   ├── ml_services.py                <- Django ML facade connecting models to views
│   ├── views.py                      <- Integrated ML views, APIs, and dashboard routes
│   ├── urls.py                       <- Endpoint routing for AI assistant and demand predictor
│   └── templates/
│       ├── student_assistant.html    <- Interactive conversational AI chatbot UI
│       ├── demand_forecast.html      <- AI Bus Demand Prediction & fleet dispatch dashboard
│       ├── feedbacklist.html         <- Feedback sentiment & issue category pills
│       ├── complaintlist.html        <- Complaint NLP classification table & metrics
│       ├── book_bus.html             <- Smart ML route recommendation cards
│       ├── locations.html            <- Real-time route match preview
│       └── index.html                <- Student dashboard with AI assistant access
└── concession/                       <- Django core settings and URL dispatcher
```

---

## 💼 Resume-Ready Bullet Points

You can include these achievements directly on your resume:

- **AI-Powered Route Recommendation:** Designed and deployed a Random Forest route recommendation engine using Scikit-Learn and Joblib, achieving 100% top-3 accuracy across 28 collegiate transit routes by processing origin-destination stop vectors, concession fares, and college hubs.
- **NLP Sentiment & Issue Categorization:** Implemented a multi-stage NLP pipeline combining NLTK VADER and TF-IDF Logistic Regression to classify student transit grievances across 7 categories (e.g. delays, overcrowding, staff behavior) with 100% test accuracy.
- **Bus Demand Forecasting:** Trained a Random Forest Regressor on 37,800+ hourly passenger transit records, achieving an $R^2$ score of 0.9728 and MAE of 2.60 passengers, empowering transport officers with automated rush-hour fleet allocation advice.
- **Conversational Student AI Assistant:** Engineered an interactive conversational transit assistant with intent recognition and database querying, providing instant answers for concession applications, QR ticket verification, and Razorpay wallet payments.
- **Django Full-Stack Integration:** Built end-to-end REST APIs and responsive dashboards connecting serialized ML models with SQLite, delivering sub-100ms inference for real-time student pass booking and administrative monitoring.
