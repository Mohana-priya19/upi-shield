import pandas as pd
from sklearn.model_selection import train_test_split
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report
import os

# Load engineered dataset
df = pd.read_csv(r'C:\Users\LENOVO\Downloads\upi-shield\data\processed\featured_transactions.csv')

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

reference = X_train.sample(5000, random_state=42)
current = X_test.sample(1000, random_state=42)

definition = DataDefinition(numerical_columns=FEATURES)

reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
current_dataset = Dataset.from_pandas(current, data_definition=definition)

report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(reference_dataset, current_dataset)

os.makedirs(r'C:\Users\LENOVO\Downloads\upi-shield\reports', exist_ok=True)
output_path = r'C:\Users\LENOVO\Downloads\upi-shield\reports\drift_report.html'
my_eval.save_html(output_path)

print(f"Drift report saved: {output_path}")
print("Open drift_report.html in your browser to view.")