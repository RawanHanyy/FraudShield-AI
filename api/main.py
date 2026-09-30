from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import io

MODEL_VERSION = "1.0"

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
    "threshold": float(threshold),
    "model_version": MODEL_VERSION
    }

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    contents = await file.read()

    try:
        data = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read CSV file."
        )

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_columns}"
        )

    data = data[required_columns]

    processed = preprocessor.transform(data)

    probabilities = model.predict_proba(processed)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    def get_risk(probability):
        if probability < 0.30:
            return "Low"
        elif probability < threshold:
            return "Medium"
        else:
            return "High"

    results = data.copy()

    results["fraud_probability"] = probabilities.round(4)
    results["prediction"] = predictions
    results["risk_level"] = [
        get_risk(p) for p in probabilities
    ]

    return {
        "model_version": MODEL_VERSION,
        "threshold": float(threshold),
        "number_of_transactions": len(results),
        "results": results.to_dict(orient="records")
    }