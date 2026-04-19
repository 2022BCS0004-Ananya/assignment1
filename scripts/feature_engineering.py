from datetime import datetime, timedelta
import numpy as np

def extract_features(customer: dict) -> dict:
    """Extract ML features from a customer record"""
    
    tickets = customer.get("tickets", [])
    now = datetime.now()
    
    def count_tickets_in_days(days):
        cutoff = now - timedelta(days=days)
        return sum(
            1 for t in tickets
            if datetime.fromisoformat(t["date"]) > cutoff
        )
    
    freq_7d = count_tickets_in_days(7)
    freq_30d = count_tickets_in_days(30)
    freq_90d = count_tickets_in_days(90)
    
    complaint_count = sum(1 for t in tickets if t["type"] == "complaint")
    
    # Average gap between tickets (in days)
    if len(tickets) >= 2:
        dates = sorted([datetime.fromisoformat(t["date"]) for t in tickets])
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_gap = np.mean(gaps)
    else:
        avg_gap = 0.0
    
    charge_diff = customer.get("monthly_charges", 0) - customer.get("previous_month_charges", 0)
    
    return {
        "freq_7d": freq_7d,
        "freq_30d": freq_30d,
        "freq_90d": freq_90d,
        "complaint_count": complaint_count,
        "avg_gap": round(float(avg_gap), 2),
        "charge_diff": round(float(charge_diff), 2)
    }