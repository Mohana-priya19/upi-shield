from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="UPI Shield Fraud Detection API")

# Load model once at startup
model = joblib.load(r'C:\Users\LENOVO\Downloads\upi-shield\models\xgboost_baseline.pkl')

FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'hour', 'day', 'is_weekend', 'type_encoded',
    'balance_diff_orig', 'balance_diff_dest',
    'orig_balance_zero', 'amount_deviation'
]

# Input schema
class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    hour: int
    day: int
    is_weekend: int
    type_encoded: int  # 1=TRANSFER, 0=CASH_OUT

@app.get("/")
def root():
    return {"message": "UPI Shield Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Engineer features from input
    balance_diff_orig = transaction.oldbalanceOrg - transaction.newbalanceOrig
    balance_diff_dest = transaction.newbalanceDest - transaction.oldbalanceDest
    orig_balance_zero = 1 if transaction.newbalanceOrig == 0 else 0

    type_mean = 910647.0 if transaction.type_encoded == 1 else 186258.0
    amount_deviation = transaction.amount - type_mean

    # Build feature row
    features = pd.DataFrame([{
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'newbalanceOrig': transaction.newbalanceOrig,
        'oldbalanceDest': transaction.oldbalanceDest,
        'newbalanceDest': transaction.newbalanceDest,
        'hour': transaction.hour,
        'day': transaction.day,
        'is_weekend': transaction.is_weekend,
        'type_encoded': transaction.type_encoded,
        'balance_diff_orig': balance_diff_orig,
        'balance_diff_dest': balance_diff_dest,
        'orig_balance_zero': orig_balance_zero,
        'amount_deviation': amount_deviation
    }])

    # Predict
    fraud_proba = model.predict_proba(features)[0][1]
    is_fraud = bool(fraud_proba >= 0.5)

    return {
        "fraud_detected": is_fraud,
        "fraud_probability": round(float(fraud_proba), 4),
        "risk_level": "HIGH" if fraud_proba >= 0.7 else "MEDIUM" if fraud_proba >= 0.3 else "LOW",
        "verdict": "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE TRANSACTION"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost_baseline"}