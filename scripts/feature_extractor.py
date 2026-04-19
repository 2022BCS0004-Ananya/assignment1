from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        now = datetime.now()

        for customer in X:
            tickets = customer.get("tickets", [])

            def parse_date(date_str):
                try:
                    return datetime.fromisoformat(date_str)
                except:
                    return now - timedelta(days=999)

            def count_in_days(days):
                cutoff = now - timedelta(days=days)
                return sum(1 for t in tickets if parse_date(t["date"]) > cutoff)

            freq_7d  = count_in_days(7)
            freq_30d = count_in_days(30)
            freq_90d = count_in_days(90)
            complaint_count = sum(1 for t in tickets if t["type"] == "complaint")

            if len(tickets) >= 2:
                dates = sorted([parse_date(t["date"]) for t in tickets])
                gaps  = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
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