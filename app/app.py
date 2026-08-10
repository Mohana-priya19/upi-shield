from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import sys
sys.path.append(r'C:\Users\LENOVO\Downloads\upi-shield')

app = Flask(__name__)

model = joblib.load(r'C:\Users\LENOVO\Downloads\upi-shield\models\xgboost_baseline.pkl')

FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'hour', 'day', 'is_weekend', 'type_encoded',
    'balance_diff_orig', 'balance_diff_dest',
    'orig_balance_zero', 'amount_deviation'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    amount = float(data['amount'])
    oldbalanceOrg = float(data['oldbalanceOrg'])
    newbalanceOrig = float(data['newbalanceOrig'])
    oldbalanceDest = float(data['oldbalanceDest'])
    newbalanceDest = float(data['newbalanceDest'])
    hour = int(data['hour'])
    day = int(data['day'])
    is_weekend = int(data['is_weekend'])
    type_encoded = int(data['type_encoded'])

    balance_diff_orig = oldbalanceOrg - newbalanceOrig
    balance_diff_dest = newbalanceDest - oldbalanceDest
    orig_balance_zero = 1 if newbalanceOrig == 0 else 0
    type_mean = 910647.0 if type_encoded == 1 else 186258.0
    amount_deviation = amount - type_mean

    features = pd.DataFrame([{
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'hour': hour,
        'day': day,
        'is_weekend': is_weekend,
        'type_encoded': type_encoded,
        'balance_diff_orig': balance_diff_orig,
        'balance_diff_dest': balance_diff_dest,
        'orig_balance_zero': orig_balance_zero,
        'amount_deviation': amount_deviation
    }])

    fraud_proba = model.predict_proba(features)[0][1]
    is_fraud = bool(fraud_proba >= 0.5)
    risk = "HIGH" if fraud_proba >= 0.7 else "MEDIUM" if fraud_proba >= 0.3 else "LOW"

    return jsonify({
        'fraud_detected': is_fraud,
        'fraud_probability': round(float(fraud_proba) * 100, 2),
        'risk_level': risk,
        'verdict': 'FRAUD DETECTED' if is_fraud else 'LEGITIMATE TRANSACTION'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)