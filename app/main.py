from fastapi import FastAPI
from app.models import Customer
from app.rules import calculate_risk
from app.data_loader import load_data
from app.logger import get_logger

import joblib
import os
import time
import psutil
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="Churn Prediction ML Service", version="0.1.0")
logger = get_logger(__name__)

data = load_data()

MODEL_PATH = "model/churn_pipeline.pkl"
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None and os.path.exists(MODEL_PATH):
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def extract_features(customer_dict: dict) -> dict:
    tickets = customer_dict.get("tickets", [])
    now = datetime.now()

    def count_in_days(days):
        cutoff = now - timedelta(days=days)
        return sum(
            1 for t in tickets
            if datetime.fromisoformat(t["date"]) > cutoff
        )

    freq_7d  = count_in_days(7)
    freq_30d = count_in_days(30)
    freq_90d = count_in_days(90)
    complaint_count = sum(1 for t in tickets if t["type"] == "complaint")

    if len(tickets) >= 2:
        dates = sorted([datetime.fromisoformat(t["date"]) for t in tickets])
        gaps  = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_gap = float(np.mean(gaps))
    else:
        avg_gap = 0.0

    charge_diff = float(
        customer_dict.get("monthly_charges", 0) -
        customer_dict.get("previous_month_charges", 0)
    )

    return {
        "freq_7d":          freq_7d,
        "freq_30d":         freq_30d,
        "freq_90d":         freq_90d,
        "complaint_count":  complaint_count,
        "avg_gap":          round(avg_gap, 2),
        "charge_diff":      round(charge_diff, 2)
    }


@app.get("/")
def home():
    return {"message": "Churn Prediction ML Service is running"}


@app.post("/predict-risk")
def predict_risk(customer: Customer):
    start = time.time()

    # Convert Pydantic model → plain dict
    customer_dict = {
        "monthly_charges":          customer.monthly_charges,
        "previous_month_charges":   customer.previous_month_charges,
        "contract_type":            customer.contract_type,
        "tickets": [
            {"type": t.type, "date": t.date}
            for t in customer.tickets
        ]
    }

    features = extract_features(customer_dict)
    model    = get_pipeline()

    if model:
        label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        pred = model.predict([customer_dict])[0]
        risk = label_map.get(int(pred), "LOW")
    else:
        # Fallback to original rule engine
        risk = calculate_risk(customer_dict)

    latency = time.time() - start
    logger.info(f"Prediction: {risk} | Latency: {latency:.4f}s")

    return {
        "risk":             risk,
        "features_used":    features,
        "latency_seconds":  round(latency, 6)
    }


@app.get("/customers")
def get_customers():
    return data[:10]


@app.get("/metrics")
def get_metrics():
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "memory_usage_mb":    round(mem.rss / 1024 / 1024, 2),
        "cpu_usage_percent":  psutil.cpu_percent(interval=0.1)
    }