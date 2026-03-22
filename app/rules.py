from datetime import datetime, timedelta

def calculate_risk(data):
    tickets = data["tickets"]
    now = datetime.now()

    recent_tickets = [
        t for t in tickets
        if datetime.fromisoformat(t["date"]) > now - timedelta(days=30)
    ]

    # Rule 1
    if len(recent_tickets) > 5:
        return "HIGH"

    # Rule 2
    if (data["monthly_charges"] > data["previous_month_charges"]) and len(tickets) >= 3:
        return "MEDIUM"

    # Rule 3
    if data["contract_type"] == "Month-to-Month":
        if any(t["type"] == "complaint" for t in tickets):
            return "HIGH"

    return "LOW"