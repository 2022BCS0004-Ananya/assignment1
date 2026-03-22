import json
import os

def load_data():
    path = "data/processed_data.json"

    if not os.path.exists(path):
        raise Exception("processed_data.json not found. Run preprocess.py")

    if os.path.getsize(path) == 0:
        raise Exception("processed_data.json is empty. Regenerate it")

    with open(path) as f:
        return json.load(f)