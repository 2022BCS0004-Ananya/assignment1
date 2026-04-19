import json
import joblib
import numpy as np
import os
import sys

# Make sure app/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)
from datetime import datetime, timedelta


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts numerical features from raw customer dicts"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        now = datetime.now()

        for customer in X:
            tickets = customer.get("tickets", [])

            def count_in_days(days):
                cutoff = now - timedelta(days=days)
                return sum(
                    1 for t in tickets
                    if datetime.fromisoformat(t["date"]) > cutoff
                )

            freq_7d  = count_in_days(7)
            freq_30d = count_in_days(30)
            freq_90d = count_in_days(90)
            complaint_count = sum(1 for t in tickets if t["type"] == "complaint")

            if len(tickets) >= 2:
                dates = sorted([datetime.fromisoformat(t["date"]) for t in tickets])
                gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                avg_gap = float(np.mean(gaps))
            else:
                avg_gap = 0.0

            charge_diff = float(
                customer.get("monthly_charges", 0) -
                customer.get("previous_month_charges", 0)
            )

            rows.append([freq_7d, freq_30d, freq_90d,
                         complaint_count, avg_gap, charge_diff])

        return np.array(rows)


def assign_label(customer):
    """Rule-based label for training (LOW=0, MEDIUM=1, HIGH=2)"""
    tickets = customer.get("tickets", [])
    complaint_count = sum(1 for t in tickets if t["type"] == "complaint")
    monthly  = customer.get("monthly_charges", 0)
    previous = customer.get("previous_month_charges", 0)
    churn    = customer.get("churn", False)

    if churn or complaint_count >= 5:
        return 2  # HIGH
    elif monthly > previous and len(tickets) >= 3:
        return 1  # MEDIUM
    else:
        return 0  # LOW


def train():
    data_path = "data/processed/processed_data.json"
    if not os.path.exists(data_path):
        data_path = "data/processed_data.json"  # fallback to your old path

    with open(data_path) as f:
        customers = json.load(f)

    labels = [assign_label(c) for c in customers]

    X_train, X_test, y_train, y_test = train_test_split(
        customers, labels, test_size=0.2, random_state=42
    )

    # Save splits
    os.makedirs("data/splits", exist_ok=True)
    with open("data/splits/X_train.json", "w") as f:
        json.dump(X_train, f)
    with open("data/splits/X_test.json", "w") as f:
        json.dump(X_test, f)
    with open("data/splits/y_train.json", "w") as f:
        json.dump(y_train, f)
    with open("data/splits/y_test.json", "w") as f:
        json.dump(y_test, f)
    print("✅ Train-test splits saved!")

    n_estimators = 100
    max_depth    = None

    mlflow.set_tracking_uri("sqlite:///mlflow_docker.db")  # fresh DB inside container
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        mlflow.log_param("n_estimators",    n_estimators)
        mlflow.log_param("max_depth",       str(max_depth))
        mlflow.log_param("dataset_version", "v1")
        mlflow.log_param("features", str([
            "freq_7d", "freq_30d", "freq_90d",
            "complaint_count", "avg_gap", "charge_diff"
        ]))

        pipeline = Pipeline([
            ("features", FeatureExtractor()),
            ("model",    RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_test, y_pred,    average="macro", zero_division=0)
        f1        = f1_score(y_test, y_pred,        average="macro", zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob,   multi_class="ovr")

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"\n📊 Metrics:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        print(f"ROC-AUC:   {roc_auc:.2f}")

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)

        os.makedirs("model", exist_ok=True)
        joblib.dump(pipeline, "model/churn_pipeline.pkl")
        joblib.dump(pipeline, "model/churn_model.pkl")

        with open("model/metrics.txt", "w") as f:
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"ROC-AUC: {roc_auc}\n")

        try:
            mlflow.sklearn.log_model(
                pipeline,
                "model1",
                registered_model_name="ChurnPredictionModel"
            )
        except Exception as e:
            print(f"MLflow model registry warning: {e}")

        print("✅ Model trained, tracked, and saved successfully!")


if __name__ == "__main__":
    train()