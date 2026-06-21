import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_PATH = "PS_20174392719_1491204439457_log.csv"
MODEL_PATH = "fraud_detection_model.pkl"
RESULTS_FILE = "fraud_results.csv"


def preprocess_logic(df):
    df = df.copy()
    df['diff_balance_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['diff_balance_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['origin_balance_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1)
    df['dest_balance_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1)
    df['amount_to_old_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    all_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    for t in all_types:
        col_name = f'type_{t}'
        df[col_name] = (df['type'] == t).astype(int) if 'type' in df.columns else 0

    return df


@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        saved_data = joblib.load(MODEL_PATH)
        return saved_data['model'], saved_data['features'], saved_data.get('metrics', None)
    else:
        if not os.path.exists(DATA_PATH):
            st.error(f"File not found: {DATA_PATH}")
            return None, None, None

        with st.spinner("Training Model..."):
            df = pd.read_csv(DATA_PATH)
            df = preprocess_logic(df)

            type_cols = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
            features = ['amount', 'diff_balance_org', 'diff_balance_dest',
                        'origin_balance_ratio', 'dest_balance_ratio',
                        'amount_to_old_ratio'] + type_cols

            X = df[features]
            y = df['isFraud']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model = LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1)
            model.fit(X_train, y_train)

            # ✅ حساب الـ metrics
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            metrics = {'accuracy': acc, 'f1': f1, 'auc': auc}

            print(f"\n{'='*40}")
            print(f"Accuracy  : {acc:.2%}")
            print(f"F1 Score  : {f1:.2%}")
            print(f"AUC Score : {auc:.2%}")
            print(f"{'='*40}\n")

            model_data = {'model': model, 'features': features, 'metrics': metrics}
            joblib.dump(model_data, MODEL_PATH)
            return model, features, metrics


st.set_page_config(page_title="Fraud Guard Pro", layout="wide")
st.title("🛡️ Real-Time Fraud Detection System")

ensemble, feature_names, metrics = load_or_train_model()

if ensemble:
    # ✅ عرض الـ metrics في الـ app
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("F1 Score", f"{metrics['f1']:.2%}")
        col3.metric("AUC Score", f"{metrics['auc']:.2%}")

    st.sidebar.header("System Controls")
    threshold = st.sidebar.slider("Fraud Sensitivity", 0.05, 0.95, 0.50)

    if st.sidebar.button("Clear Activity Logs"):
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Transaction Entry")

    with st.sidebar.form("input_form"):
        type_tx = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)
        old_org = st.number_input("Current Balance", min_value=0.0, value=5000.0)
        old_dest = st.number_input("Recipient Balance", min_value=0.0, value=0.0)

        if type_tx in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
            new_org = old_org - amount
            new_dest = old_dest + amount
        else:
            new_org = old_org + amount
            new_dest = max(0.0, old_dest - amount)

        st.info(f"Calculated New Balance: {new_org}")
        submit = st.form_submit_button("Analyze Transaction 🔍")

    if submit:
        new_tx_data = pd.DataFrame([{
            'type': type_tx, 'amount': amount,
            'oldbalanceOrg': old_org, 'newbalanceOrig': new_org,
            'oldbalanceDest': old_dest, 'newbalanceDest': new_dest
        }])

        df_processed = preprocess_logic(new_tx_data)
        final_input = df_processed[feature_names]

        prob = ensemble.predict_proba(final_input)[:, 1][0]
        is_fraud = 1 if prob >= threshold else 0

        log_entry = pd.DataFrame({
            'Time': [datetime.now().strftime("%H:%M:%S")],
            'Type': [type_tx], 'Amount': [amount],
            'Prob': [f"{prob:.2%}"],
            'Status': ['FRAUD' if is_fraud == 1 else 'SAFE']
        })
        log_entry.to_csv(RESULTS_FILE, mode='a', index=False, header=not os.path.exists(RESULTS_FILE))

        if is_fraud == 1:
            st.error(f"🚨 FRAUD DETECTED! Probability: {prob:.2%}")
        else:
            st.success(f"✅ Transaction Verified. Probability: {prob:.2%}")

    st.divider()
    if os.path.exists(RESULTS_FILE):
        df_history = pd.read_csv(RESULTS_FILE)
        st.subheader("📋 Recent Activity Log")
        st.dataframe(df_history.iloc[::-1], use_container_width=True)
