#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL
# =========================
model = joblib.load("churn_pipeline.pkl")

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# =========================
# TITLE
# =========================
st.title("📊 Customer Churn Analytics Dashboard")

st.markdown("End-to-end ML system for predicting and analyzing customer churn.")

# =========================
# LOAD SAMPLE DATA (for dashboard visuals)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    df = df.dropna()
    return df

df = load_data()

# =========================
# KPI SECTION
# =========================
total_customers = len(df)
churn_rate = df["Churn"].mean() * 100
non_churn_rate = 100 - churn_rate

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Retention Rate", f"{non_churn_rate:.2f}%")

st.divider()

# =========================
# VISUALIZATION SECTION
# =========================
st.subheader("📈 Churn Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📌 Feature Importance (Model Insight)")

if hasattr(model.named_steps["model"], "feature_importances_"):
    importance = model.named_steps["model"].feature_importances_

    # Approx labels (important: pipeline transforms features)
    feature_names = df.drop("Churn", axis=1).columns[:len(importance)]

    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=feature_names, ax=ax)
    st.pyplot(fig)

else:
    st.info("Feature importance not available")

st.divider()

# =========================
# PREDICTION SECTION
# =========================
st.subheader("🧠 Predict Customer Churn")

with st.sidebar:
    st.header("Customer Input")

    support_calls = st.number_input("Support Calls")
    total_spend = st.number_input("Total Spend")
    age = st.number_input("Age")
    payment_delay = st.number_input("Payment Delay")
    last_interaction = st.number_input("Last Interaction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    contract = st.selectbox("Contract Length", ["Monthly", "Yearly", "Two Year"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

    predict_btn = st.button("Predict")

if predict_btn:

    input_df = pd.DataFrame([{
        "Support Calls": support_calls,
        "Total Spend": total_spend,
        "Age": age,
        "Payment Delay": payment_delay,
        "Last Interaction": last_interaction,
        "Gender": gender,
        "Contract Length": contract,
        "Subscription Type": subscription
    }])

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ Customer WILL CHURN (Risk: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Risk: {prob:.2f})")

