"""
Data Preparation & Cleaning Pipeline for GoTap AI/ML
===================================================
Uses data cleaning functions from ML_AI_PROJECT/data/clean_data.py:
- Cleans stop names with alias resolution
- Cleans NLP text (special characters, noise removal, normalization)
- Validates features and imputes missing numeric values
- Saves clean processed datasets to data/processed/
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure parent is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.clean_data import (
    clean_stop_name,
    clean_nlp_text,
    validate_route_dataframe,
    validate_demand_dataframe
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_route_data():
    """Clean and prepare route recommendation dataset."""
    raw_path = os.path.join(RAW_DIR, "route_travel_dataset.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw data: {raw_path}")
    
    df = pd.read_csv(raw_path)
    print(f"Cleaning Route Travel Data (initial: {len(df)} rows)...")
    
    # Clean stops
    df['origin_stop_clean'] = df['origin_stop'].apply(clean_stop_name)
    df['destination_stop_clean'] = df['destination_stop'].apply(clean_stop_name)
    
    # Feature engineering: route direction flag
    df['is_kerala_varma_origin'] = df['origin_stop_clean'].str.contains('Kerala Varma', case=False, na=False).astype(int)
    df['is_vimala_origin'] = df['origin_stop_clean'].str.contains('Vimala', case=False, na=False).astype(int)
    
    # Ensure numeric columns
    df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
    df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
    df['travel_hour'] = pd.to_numeric(df['travel_hour'], errors='coerce').fillna(8).astype(int)
    df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
    
    # Clean duplicates & missing
    df = df.dropna(subset=['target_route_id', 'origin_stop_clean', 'destination_stop_clean'])
    df = df.drop_duplicates()
    
    out_path = os.path.join(PROCESSED_DIR, "route_travel_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned Route Travel Data: {len(df)} rows saved to {out_path}")
    return df


def clean_demand_data():
    """Clean and validate historical bus demand dataset."""
    raw_path = os.path.join(RAW_DIR, "historical_bus_demand.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw data: {raw_path}")
        
    df = pd.read_csv(raw_path)
    print(f"Cleaning Historical Demand Data (initial: {len(df)} rows)...")
    
    df = validate_demand_dataframe(df)
    
    # Feature engineering: rush hour classification
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    
    out_path = os.path.join(PROCESSED_DIR, "bus_demand_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned Historical Demand Data: {len(df)} rows saved to {out_path}")
    return df


def clean_nlp_data():
    """Clean and prepare NLP feedback and complaint corpus."""
    raw_path = os.path.join(RAW_DIR, "feedback_complaint_nlp.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw data: {raw_path}")
        
    df = pd.read_csv(raw_path)
    print(f"Cleaning NLP Feedback/Complaint Data (initial: {len(df)} rows)...")
    
    df['clean_text'] = df['text'].apply(clean_nlp_text)
    # Filter empty or very short strings
    df = df[df['clean_text'].str.len() > 8]
    df = df.drop_duplicates(subset=['clean_text', 'issue_category'])
    
    out_path = os.path.join(PROCESSED_DIR, "feedback_complaint_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned NLP Data: {len(df)} rows saved to {out_path}")
    return df


if __name__ == "__main__":
    print("Executing GoTap Data Preparation Pipeline...")
    clean_route_data()
    clean_demand_data()
    clean_nlp_data()
    print("All processed datasets prepared successfully!")
