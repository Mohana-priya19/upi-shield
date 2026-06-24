import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw PaySim CSV and filter to fraud-relevant transaction types."""
    df = pd.read_csv(filepath)
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    print(f"Loaded {len(df):,} transactions ({df['isFraud'].sum():,} fraud)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 7 features from raw PaySim data.

    Features added:
        hour              - Hour of day (0-23) derived from step
        day               - Day number derived from step
        is_weekend        - Proxy weekend flag (day % 7)
        type_encoded      - 1 if TRANSFER, 0 if CASH_OUT
        balance_diff_orig - Amount drained from origin account
        balance_diff_dest - Amount gained by destination account
        orig_balance_zero - 1 if origin account drained to zero
        amount_deviation  - Deviation from per-type mean transaction amount
    """
    df = df.copy()

    # Time features
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24
    df['is_weekend'] = df['day'] % 7

    # Transaction type encoding
    df['type_encoded'] = (df['type'] == 'TRANSFER').astype(int)

    # Balance difference features
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Origin account drained to zero
    df['orig_balance_zero'] = (df['newbalanceOrig'] == 0).astype(int)

    # Amount deviation from per-type mean
    type_mean = df.groupby('type')['amount'].transform('mean')
    df['amount_deviation'] = df['amount'] - type_mean

    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save engineered dataset to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} — shape: {df.shape}")


if __name__ == "__main__":
    RAW_PATH = r'C:\Users\LENOVO\Downloads\upi-shield\data\raw\paysim.csv.csv'
    OUT_PATH = r'C:\Users\LENOVO\Downloads\upi-shield\data\processed\featured_transactions.csv'

    df = load_raw_data(RAW_PATH)
    df = engineer_features(df)
    save_processed_data(df, OUT_PATH)