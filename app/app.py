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
        
st.divider()

st.subheader("📁 Batch Transaction Analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transactions",
    type=["csv"]
)

if uploaded_file is not None:

    batch_data = pd.read_csv(uploaded_file)

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in batch_data.columns
    ]

    if missing_columns:
        st.error(
            f"Missing required columns: {missing_columns}"
        )

    else:

        batch_input = batch_data[required_columns]

        batch_processed = preprocessor.transform(batch_input)

        batch_probabilities = model.predict_proba(
            batch_processed
        )[:, 1]

        batch_predictions = (
            batch_probabilities >= threshold
        ).astype(int)

        def get_risk(probability):
            if probability < 0.30:
                return "Low"
            elif probability < threshold:
                return "Medium"
            else:
                return "High"

        results = batch_input.copy()

        results["Fraud Probability"] = batch_probabilities
        results["Prediction"] = batch_predictions
        results["Risk Level"] = [
            get_risk(p)
            for p in batch_probabilities
        ]

        st.success(
            f"Analyzed {len(results)} transactions."
        )

        st.dataframe(results)

        high_risk_count = (
            results["Risk Level"] == "High"
        ).sum()

        st.metric(
            "High-Risk Transactions",
            int(high_risk_count)
        )