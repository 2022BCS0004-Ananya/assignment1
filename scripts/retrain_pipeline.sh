#!/bin/bash
echo "#01 Checking data drift..."
python scripts/check_drift.py

echo "#02 Retraining model..."
python scripts/train_model.py

echo "✅ Model retrained and registered successfully!"