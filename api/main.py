from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="FraudShield AI API",
    description="Machine learning API for transaction fraud-risk scoring.",
    version="1.0"
)

preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/fraud_model.joblib")
threshold = joblib.load("models/threshold.joblib")


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float


@app.get("/")
def home():
    return {"message": "FraudShield AI API is running"}


@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([transaction.model_dump()])

    processed = preprocessor.transform(data)

    fraud_probability = model.predict_proba(processed)[:, 1][0]

    prediction = int(fraud_probability >= threshold)

    if fraud_probability < 0.30:
        risk_level = "Low"
    elif fraud_probability < threshold:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": float(threshold)
    }
