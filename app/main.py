from fastapi import FastAPI
from app.rules import calculate_risk
from app.data_loader import load_data

app = FastAPI()

data = load_data()

@app.get("/")
def home():
    return {"message": "Churn Risk Service Running"}

@app.post("/predict-risk")
def predict_risk(customer: dict):
    risk = calculate_risk(customer)
    return {"risk": risk}

@app.get("/customers")
def get_customers():
    return data[:10]  # sample