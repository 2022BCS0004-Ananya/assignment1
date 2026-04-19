import pandas as pd
import random
from datetime import datetime, timedelta
import json
import os

def generate_processed_data():
    raw_path = "data/Telco-Customer-Churn.csv"
    
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        df = df[["customerID", "MonthlyCharges", "Contract"]]
        df.rename(columns={
            "MonthlyCharges": "monthly_charges",
            "Contract": "contract_type"
        }, inplace=True)
        
        df["previous_month_charges"] = df["monthly_charges"].apply(
            lambda x: round(x + random.randint(-20, 20), 2)
        )
        
        # Add churn label based on contract type for ML training
        def assign_churn(row):
            if row["contract_type"] == "Month-to-month":
                return random.random() < 0.45  # 45% churn for month-to-month
            elif row["contract_type"] == "One year":
                return random.random() < 0.15
            else:
                return random.random() < 0.05
        
        df["churn"] = df.apply(assign_churn, axis=1)
        
        def generate_tickets(is_churn):
            tickets = []
            num = random.randint(5, 10) if is_churn else random.randint(0, 3)
            for _ in range(num):
                t_type = "complaint" if is_churn and random.random() > 0.3 else random.choice(["complaint", "query"])
                tickets.append({
                    "type": t_type,
                    "date": (datetime.now() - timedelta(days=random.randint(0, 60))).isoformat()
                })
            return tickets
        
        df["tickets"] = df.apply(lambda row: generate_tickets(row["churn"]), axis=1)
        
        data = df.to_dict(orient="records")
    
    else:
        # Synthetic fallback
        data = []
        now = datetime.now()
        contract_types = ["Month-to-month", "One year", "Two year"]
        
        for i in range(1409):
            is_churn = i % 3 == 0
            monthly = round(random.uniform(20, 120), 2)
            num_tickets = random.randint(5, 10) if is_churn else random.randint(0, 3)
            
            tickets = []
            for _ in range(num_tickets):
                t_type = "complaint" if is_churn else random.choice(["complaint", "query"])
                tickets.append({
                    "type": t_type,
                    "date": (now - timedelta(days=random.randint(0, 60))).isoformat()
                })
            
            data.append({
                "customerID": f"CUST-{i:04d}",
                "monthly_charges": monthly,
                "previous_month_charges": round(monthly + random.randint(-20, 20), 2),
                "contract_type": random.choice(contract_types),
                "tickets": tickets,
                "churn": is_churn
            })
    
    # Save to BOTH paths so existing code doesn't break
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    with open("data/processed/processed_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Processed data saved! ({len(data)} records)")
    return data

if __name__ == "__main__":
    generate_processed_data()