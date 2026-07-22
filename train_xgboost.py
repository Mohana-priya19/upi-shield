import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

# MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("upi-shield-xgboost")

# Load the engineered dataset (already preprocessed in Day 4)
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\upi-shield\data\processed\featured_transactions.csv")

FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'hour', 'day', 'is_weekend', 'type_encoded',
    'balance_diff_orig', 'balance_diff_dest',
    'orig_balance_zero', 'amount_deviation'
]

X = df[FEATURES]
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": round(scale, 1),
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1
}

with mlflow.start_run(run_name="xgb_baseline_v1"):
    mlflow.log_params(params)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("fraud_recall", recall)

    mlflow.xgboost.log_model(model, "model")

    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"Recall:   {recall:.4f}")
    print("Run logged to MLflow.")