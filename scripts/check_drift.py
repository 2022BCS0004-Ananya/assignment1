import json
import numpy as np
import os

def check_drift():
    old_path = "data/processed/processed_data.json"
    new_path = "data/processed/new_data.json"
    
    if not os.path.exists(new_path):
        # Create synthetic new data to simulate drift
        with open(old_path) as f:
            old_data = json.load(f)
        
        # Simulate drift: inflate monthly charges
        new_data = []
        for c in old_data[:100]:
            new_c = c.copy()
            new_c["monthly_charges"] = c["monthly_charges"] + 225
            new_data.append(new_c)
        
        with open(new_path, "w") as f:
            json.dump(new_data, f)
    
    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)
    
    old_mean = np.mean([c["monthly_charges"] for c in old_data])
    new_mean = np.mean([c["monthly_charges"] for c in new_data])
    drift = abs(old_mean - new_mean)
    
    print(f"Old Mean: {old_mean}")
    print(f"New Mean: {new_mean}")
    print(f"Drift: {drift}")
    
    if drift > 10:
        print("⚠ Drift detected!")
        return True
    else:
        print("✅ No significant drift.")
        return False

if __name__ == "__main__":
    check_drift()