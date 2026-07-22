import sys
sys.path.append(r'C:\Users\LENOVO\Downloads\upi-shield')

import pandas as pd
import numpy as np
import joblib
from src.features import engineer_features

# ── Test 1: Feature engineering output shape ──────────────────────────────────
def test_feature_engineering_columns():
    raw = pd.DataFrame([{
        'step': 1, 'type': 'TRANSFER', 'amount': 1000,
        'nameOrig': 'A', 'oldbalanceOrg': 1000, 'newbalanceOrig': 0,
        'nameDest': 'B', 'oldbalanceDest': 0, 'newbalanceDest': 1000,
        'isFraud': 1, 'isFlaggedFraud': 0
    }])
    result = engineer_features(raw)
    expected = ['hour','day','is_weekend','type_encoded',
                'balance_diff_orig','balance_diff_dest',
                'orig_balance_zero','amount_deviation']
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"
    print("✅ Test 1 passed: all engineered features present")

# ── Test 2: orig_balance_zero flag works correctly ────────────────────────────
def test_orig_balance_zero_flag():
    raw = pd.DataFrame([{
        'step': 1, 'type': 'CASH_OUT', 'amount': 500,
        'nameOrig': 'A', 'oldbalanceOrg': 500, 'newbalanceOrig': 0,
        'nameDest': 'B', 'oldbalanceDest': 0, 'newbalanceDest': 500,
        'isFraud': 1, 'isFlaggedFraud': 0
    }])
    result = engineer_features(raw)
    assert result['orig_balance_zero'].iloc[0] == 1, "Flag should be 1 when balance drained"
    print("✅ Test 2 passed: orig_balance_zero flag correct")

# ── Test 3: Model loads and predicts correct shape ────────────────────────────
def test_model_prediction_shape():
    model = joblib.load(r'C:\Users\LENOVO\Downloads\upi-shield\models\xgboost_baseline.pkl')
    sample = pd.DataFrame([{
        'amount': 450000, 'oldbalanceOrg': 450000, 'newbalanceOrig': 0,
        'oldbalanceDest': 0, 'newbalanceDest': 450000,
        'hour': 3, 'day': 1, 'is_weekend': 0, 'type_encoded': 1,
        'balance_diff_orig': 450000, 'balance_diff_dest': 450000,
        'orig_balance_zero': 1, 'amount_deviation': 100000
    }])
    proba = model.predict_proba(sample)
    assert proba.shape == (1, 2), "Output shape should be (1, 2)"
    assert 0 <= proba[0][1] <= 1, "Probability must be between 0 and 1"
    print(f"✅ Test 3 passed: model predicts fraud probability {proba[0][1]:.4f}")

# ── Test 4: Fraud transaction scores high ─────────────────────────────────────
def test_fraud_transaction_high_score():
    model = joblib.load(r'C:\Users\LENOVO\Downloads\upi-shield\models\xgboost_baseline.pkl')
    fraud = pd.DataFrame([{
        'amount': 450000, 'oldbalanceOrg': 450000, 'newbalanceOrig': 0,
        'oldbalanceDest': 0, 'newbalanceDest': 450000,
        'hour': 3, 'day': 1, 'is_weekend': 0, 'type_encoded': 1,
        'balance_diff_orig': 450000, 'balance_diff_dest': 450000,
        'orig_balance_zero': 1, 'amount_deviation': 100000
    }])
    proba = model.predict_proba(fraud)[0][1]
    assert proba > 0.5, f"Fraud transaction should score above 0.5, got {proba}"
    print(f"✅ Test 4 passed: fraud transaction scores {proba:.4f}")

if __name__ == "__main__":
    test_feature_engineering_columns()
    test_orig_balance_zero_flag()
    test_model_prediction_shape()
    test_fraud_transaction_high_score()
    print("\n✅ All 4 tests passed.")