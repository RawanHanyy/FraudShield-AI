# 🛡️ FraudShield AI

FraudShield AI is an end-to-end machine learning system for intelligent transaction fraud detection.

The project focuses on real-world fraud detection challenges including severe class imbalance, false-positive/false-negative trade-offs, threshold optimization, model comparison, and deployment.

---

## 🎯 Project Objective

The goal is to predict whether a financial transaction is fraudulent and return:

- Fraud probability
- Fraud / legitimate prediction
- Risk level: Low, Medium, or High

The system is designed as a decision-support tool rather than an automatic transaction-blocking system.

---

## 📊 Dataset

The project uses the PaySim synthetic financial transaction dataset.

The original dataset contains approximately:

- 6.36 million transactions
- 11 original columns
- Highly imbalanced fraud labels

The raw dataset is not included in this repository because of its size.

---

## 🧠 Machine Learning Workflow

The project follows the following workflow:

1. Data audit and quality checks
2. Leakage analysis
3. Feature selection
4. Stratified train / validation / test splitting
5. Numerical scaling and categorical encoding
6. Logistic Regression baseline
7. Decision Tree comparison
8. Random Forest comparison
9. Class imbalance experiments
10. Threshold tuning
11. Final test evaluation
12. Error analysis
13. Model serialization
14. FastAPI inference endpoint
15. Streamlit demonstration interface

---

## ⚙️ Features Used

The final model uses:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `oldbalanceDest`

The following fields were excluded from the initial model because they are identifiers, target-related fields, or potential leakage risks:

- `nameOrig`
- `nameDest`
- `isFlaggedFraud`
- `newbalanceOrig`
- `newbalanceDest`

---

## 🤖 Models Compared

Three supervised machine learning models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest provided the strongest validation performance and was selected as the final model.

---

## ⚖️ Class Imbalance

Fraud transactions represent only a very small proportion of the dataset.

Two imbalance-handling approaches were compared:

- Class weighting
- Random undersampling

Evaluation focused on Precision, Recall, F1-score, ROC-AUC, and PR-AUC rather than accuracy alone.

---

## 🎚️ Threshold Optimization

The default probability threshold of `0.50` was not automatically accepted.

Multiple thresholds were evaluated on the validation set.

The selected threshold was:

`0.70`

This threshold produced a better balance between fraud detection and false-positive control for the selected model.

---

## 📈 Final Test Results

Final Random Forest evaluation on the untouched test set:

- Fraud Precision: approximately **0.60**
- Fraud Recall: approximately **0.68**
- Fraud F1-score: approximately **0.64**
- ROC-AUC: approximately **0.994**
- PR-AUC: approximately **0.717**

Final confusion-matrix outcomes:

- True Negatives: 119,774
- False Positives: 71
- False Negatives: 49
- True Positives: 106

---

## 🔍 Error Analysis

The final model produced both false positives and false negatives.

False positives may increase the workload of fraud-review teams.

False negatives are more critical because they represent fraudulent transactions that remain undetected.

For this reason, the system should be used as a fraud-risk decision-support tool rather than as an automatic blocking mechanism.

---

## 🌐 FastAPI

The project includes a FastAPI service for fraud-risk inference.

Run from the repository root:

```bash
uvicorn api.main:app --reload
