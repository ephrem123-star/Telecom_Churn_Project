import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Telecom Churn Risk Demo", page_icon="📶", layout="centered")
MODEL_PATH = Path(__file__).parent / "models" / "churn_random_forest.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📶 Telecom Customer Churn Risk Demo")
st.caption("Portfolio demonstration • Fictional data • Not affiliated with or based on Safaricom customer data")
st.warning("This model was trained on synthetic data and has modest test performance. Its score is illustrative only and must not be used for real customer decisions.")

with st.form("customer_inputs"):
    st.subheader("Enter sample customer details")
    c1, c2 = st.columns(2)
    with c1:
        tenure_months = st.number_input("Tenure (months)", min_value=1, max_value=72, value=12)
        monthly_spend = st.number_input("Monthly spend (fictional currency units)", min_value=100.0, max_value=1410.0, value=650.0, step=25.0)
        data_usage_gb = st.number_input("Data usage (GB/month)", min_value=0.1, max_value=35.0, value=7.0, step=0.5)
        contract_type = st.selectbox("Contract type", ["Prepaid", "Monthly", "Annual"])
        payment_method = st.selectbox("Payment method", ["Mobile Money", "Cash/Agent", "Card", "Bank"])
    with c2:
        complaints_90d = st.number_input("Complaints in last 90 days", min_value=0, max_value=4, value=0)
        network_issue_count = st.number_input("Network issues", min_value=0, max_value=6, value=1)
        usage_change_pct = st.slider("Change in usage (%)", min_value=-67.0, max_value=70.0, value=-5.0, step=1.0)
        support_contacts_90d = st.number_input("Support contacts in last 90 days", min_value=0, max_value=5, value=1)
        region = st.selectbox("Region (fictional)", ["Central", "East", "North", "South", "West"])
    submitted = st.form_submit_button("Estimate churn risk", type="primary")

if submitted:
    row = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "data_usage_gb": data_usage_gb,
        "complaints_90d": complaints_90d,
        "network_issue_count": network_issue_count,
        "usage_change_pct": usage_change_pct,
        "support_contacts_90d": support_contacts_90d,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "region": region,
    }])
    model = load_model()
    probability = float(model.predict_proba(row)[0, 1])
    if probability < 0.30:
        segment = "Low"
    elif probability < 0.60:
        segment = "Medium"
    else:
        segment = "High"
    st.subheader("Demo result")
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.progress(probability)
    st.write(f"**Illustrative risk segment:** {segment}")
    st.info("This is a model estimate from a fictional training dataset, not a factual prediction about a real person or customer.")

st.divider()
st.caption("Built with Python, pandas, scikit-learn, joblib, and Streamlit. Synthetic data only.")
