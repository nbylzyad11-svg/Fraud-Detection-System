import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import time
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = r"C:\Users\كمبيو الكتريك\Desktop\DEPI\Real-Time Fraud Detection System\PS_20174392719_1491204439457_log.csv"
MODEL_PATH = "fraud_detection_model.pkl"
RESULTS_FILE = "fraud_results.csv"


@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        if not os.path.exists(DATA_PATH):
            st.error(f"File not found: {DATA_PATH}")
            return None

        df = pd.read_csv(DATA_PATH)
        df = preprocess_logic(df)

        type_cols = [col for col in df.columns if col.startswith('type_')]
        features = ['amount', 'diff_balance_org', 'diff_balance_dest',
                    'origin_balance_ratio', 'dest_balance_ratio',
                    'amount_to_old_ratio'] + type_cols

        X = df[features]
        y = df['isFraud']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        return model


def preprocess_logic(df):
    df = df.copy()
    df['diff_balance_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['diff_balance_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['origin_balance_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1)
    df['dest_balance_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1)
    df['amount_to_old_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    type_encoded = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_encoded], axis=1)
    return df


st.set_page_config(page_title="Fraud Guard Pro", layout="wide")
st.title("🛡️ Real-Time Fraud Detection System")

ensemble = load_or_train_model()

if ensemble:
    st.sidebar.header("System Controls")
    threshold = st.sidebar.slider("Fraud Sensitivity", 0.05, 0.95, 0.50)

    if st.sidebar.button("🗑️ Clear Activity Logs"):
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
            st.rerun()

    st.sidebar.divider()

    st.sidebar.header("Transaction Entry")
    with st.sidebar.form("input_form"):
        type_tx = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)
        old_org = st.number_input("Current Balance", min_value=0.0, value=5000.0)

        new_org = old_org - amount
        old_dest = st.number_input("Recipient Balance", min_value=0.0, value=0.0)
        new_dest = old_dest + amount

        st.info(f"Calculated New Balance: {new_org}")
        submit = st.form_submit_button("Analyze Transaction 🔍")

    if submit:
        new_tx = pd.DataFrame([{
            'step': 1, 'type': type_tx, 'amount': amount,
            'oldbalanceOrg': old_org, 'newbalanceOrig': new_org,
            'oldbalanceDest': old_dest, 'newbalanceDest': new_dest
        }])

        df_p = preprocess_logic(new_tx)
        for col in ensemble.feature_names_in_:
            if col not in df_p.columns: df_p[col] = 0

        prob = ensemble.predict_proba(df_p[ensemble.feature_names_in_])[:, 1][0]
        pred = 1 if prob >= threshold else 0

        log_entry = pd.DataFrame({
            'Time': [datetime.now().strftime("%H:%M:%S")],
            'Type': [type_tx], 'Amount': [amount],
            'Prob': [round(prob, 4)], 'Status': ['FRAUD' if pred == 1 else 'SAFE']
        })
        log_entry.to_csv(RESULTS_FILE, mode='a', index=False, header=not os.path.exists(RESULTS_FILE))

        if pred == 1:
            st.error(f"🚨 FRAUD DETECTED! Probability: {prob:.2%}")
        else:
            st.success(f"✅ Transaction Verified. Probability: {prob:.2%}")

    st.divider()

    if os.path.exists(RESULTS_FILE):
        df_history = pd.read_csv(RESULTS_FILE)
        st.subheader("📋 Recent Activity Log")
        st.dataframe(df_history.sort_index(ascending=False), use_container_width=True)

        critical_count = len(df_history[df_history['Prob'] > 0.9])
        if critical_count >= 1:
            st.error(f"🛑 CRITICAL: {critical_count} severe threats detected!")