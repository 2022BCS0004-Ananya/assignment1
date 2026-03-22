from app.rules import calculate_risk

def test_high_risk():
    data = {
        "monthly_charges": 100,
        "previous_month_charges": 80,
        "contract_type": "Month-to-Month",
        "tickets": [{"type": "complaint", "date": "2026-03-01"}]*6
    }
    assert calculate_risk(data) == "HIGH"