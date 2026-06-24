import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score)
import joblib

FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'hour', 'day', 'is_weekend', 'type_encoded',
    'balance_diff_orig', 'balance_diff_dest',
    'orig_balance_zero', 'amount_deviation'
]


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    X = df[FEATURES]
    y = df['isFraud']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_xgboost(X_train, y_train) -> XGBClassifier:
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("XGBoost trained.")
    return model


def train_isolation_forest(X_train) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=0.003,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    print("Isolation Forest trained.")
    return model


def evaluate(model, X_test, y_test, model_name: str, proba: bool = True):
    if proba:
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds = (preds_proba >= 0.5).astype(int)
        print(f"\n{'='*50}\n{model_name}\n{'='*50}")
        print(classification_report(y_test, preds, target_names=['Normal', 'Fraud']))
        print(f"ROC-AUC: {roc_auc_score(y_test, preds_proba):.4f}")
        print(f"PR-AUC:  {average_precision_score(y_test, preds_proba):.4f}")
    else:
        preds = (model.predict(X_test) == -1).astype(int)
        print(f"\n{'='*50}\n{model_name}\n{'='*50}")
        print(classification_report(y_test, preds, target_names=['Normal', 'Fraud']))
        print(f"ROC-AUC: {roc_auc_score(y_test, preds):.4f}")


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    DATA_PATH = r'C:\Users\LENOVO\Downloads\upi-shield\data\processed\featured_transactions.csv'
    MODEL_DIR = r'C:\Users\LENOVO\Downloads\upi-shield\models'

    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    xgb = train_xgboost(X_train, y_train)
    evaluate(xgb, X_test, y_test, "XGBoost")
    save_model(xgb, f"{MODEL_DIR}/xgboost_baseline.pkl")

    iso = train_isolation_forest(X_train)
    evaluate(iso, X_test, y_test, "Isolation Forest", proba=False)
    save_model(iso, f"{MODEL_DIR}/isolation_forest.pkl")