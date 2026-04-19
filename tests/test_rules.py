from datetime import datetime, timedelta
from app.rules import calculate_risk

def get_recent_date(days_ago=5):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

def test_high_risk():
    data = {
        "monthly_charges": 100,
        "previous_month_charges": 80,
        "contract_type": "Month-to-Month",
        "tickets": [{"type": "complaint", "date": get_recent_date(2)}] * 6
    }
    assert calculate_risk(data) == "HIGH"

def test_medium_risk():
    data = {
        "monthly_charges": 100,
        "previous_month_charges": 80,
        "contract_type": "One year",
        "tickets": [{"type": "query", "date": get_recent_date(5)}] * 3
    }
    result = calculate_risk(data)
    assert result in ["MEDIUM", "HIGH"]

def test_low_risk():
    data = {
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "Two year",
        "tickets": []
    }
    assert calculate_risk(data) == "LOW"