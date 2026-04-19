import json
import numpy as np

with open("data/processed/processed_data.json") as f:
    old = json.load(f)
with open("data/processed/new_data.json") as f:
    new = json.load(f)

old_charges = np.array([c["monthly_charges"] for c in old])
new_charges = np.array([c["monthly_charges"] for c in new])

drift = abs(old_charges.mean() - new_charges.mean())
print(f"Feature Drift (monthly_charges): {drift}")

if drift > 10:
    print("⚠ Feature drift detected!")
else:
    print("✅ No feature drift.")