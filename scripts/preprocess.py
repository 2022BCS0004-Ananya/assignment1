import pandas as pd
import random
from datetime import datetime, timedelta
import json

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Select required columns
df = df[["customerID", "MonthlyCharges", "Contract"]]

# Rename columns
df.rename(columns={
    "MonthlyCharges": "monthly_charges",
    "Contract": "contract_type"
}, inplace=True)

# Add previous month charges
df["previous_month_charges"] = df["monthly_charges"].apply(
    lambda x: x + random.randint(-20, 20)
)

# Generate tickets
def generate_tickets():
    tickets = []
    for _ in range(random.randint(0, 8)):
        tickets.append({
            "type": random.choice(["complaint", "query"]),
            "date": (datetime.now() - timedelta(days=random.randint(0, 60))).isoformat()
        })
    return tickets

df["tickets"] = df.apply(lambda _: generate_tickets(), axis=1)

# Convert to JSON
data = df.to_dict(orient="records")

# Save processed data
with open("data/processed_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Processed data saved!")