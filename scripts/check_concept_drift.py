import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from app.rules import calculate_risk

pipeline = joblib.load("model/churn_pipeline.pkl")

with open("data/processed/new_data.json") as f:
    new_data = json.load(f)

label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
correct = 0

for customer in new_data:
    pred_int = pipeline.predict([customer])[0]
    pred = label_map.get(int(pred_int), "LOW")
    actual = calculate_risk(customer)
    if pred == actual:
        correct += 1

accuracy = correct / len(new_data)
print(f"Accuracy on new data: {round(accuracy, 2)}")

if accuracy < 0.7:
    print("⚠ Concept drift detected!")
else:
    print("✅ Model still good!")