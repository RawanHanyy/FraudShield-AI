import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️"
)

st.title("🛡️ FraudShield AI")

st.write(
    "Machine learning system for transaction fraud-risk scoring."
)

preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/fraud_model.joblib")
threshold = joblib.load("models/threshold.joblib")

step = st.number_input("Transaction Step", min_value=1, value=1)

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
)

amount = st.number_input("Transaction Amount", min_value=0.0)

oldbalanceOrg = st.number_input(
    "Sender Balance Before Transaction",
    min_value=0.0
)

oldbalanceDest = st.number_input(
    "Receiver Balance Before Transaction",
    min_value=0.0
)

if st.button("Analyze Transaction"):

    transaction = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "oldbalanceDest": oldbalanceDest
    }])

    processed = preprocessor.transform(transaction)

    probability = model.predict_proba(processed)[:, 1][0]

    if probability < 0.30:
        risk = "Low"
    elif probability < threshold:
        risk = "Medium"
    else:
        risk = "High"

    st.metric("Fraud Probability", f"{probability:.2%}")
    st.metric("Risk Level", risk)

    if probability >= threshold:
        st.error("⚠️ Transaction flagged for fraud review.")
    else:
        st.success("Transaction is below the fraud-review threshold.")
