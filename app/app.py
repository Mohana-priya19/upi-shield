from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import sys
sys.path.append(r'C:\Users\LENOVO\Downloads\upi-shield')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'upishield2024secretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

model = joblib.load(r'C:\Users\LENOVO\Downloads\upi-shield\models\xgboost_baseline.pkl')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('about'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match. Please try again.')
            return redirect(url_for('signup'))

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash('Email already registered. Please login.')
            return redirect(url_for('login'))

        new_user = User(
            name=name,
            phone=phone,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict_page')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
@login_required
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
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)