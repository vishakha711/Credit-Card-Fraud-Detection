import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# ---- HEADER SECTION ----
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>An ML-based system to detect fraudulent transactions using XGBoost & Logistic Regression</p>", unsafe_allow_html=True)
st.write("---")

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    scaler_amount = joblib.load("scaler_amount.pkl")
    scaler_time = joblib.load("scaler_time.pkl")
    return model, scaler_amount, scaler_time

model, scaler_amount, scaler_time = load_model()

# ---- SIDEBAR INFO ----
st.sidebar.title("📊 Project Info")
st.sidebar.info("""
**Dataset:** Kaggle Credit Card Fraud (Europe, 2013)  
**Models Used:** Logistic Regression, XGBoost  
**Technique:** SMOTE Oversampling  
**Goal:** Detect fraudulent transactions  
""")
st.sidebar.write("---")
st.sidebar.markdown("[GitHub Repo](https://github.com/vishakha711/Credit-Card-Fraud-Detection)")

# ---- INPUT SECTION ----
st.subheader("🧾 Enter Transaction Details")

with st.expander("Click to Enter Transaction Features"):
    col1, col2, col3 = st.columns(3)
    input_values = []
    for i in range(1, 29):
        value = col1.number_input(f"V{i}", value=0.0, format="%.4f") if i <= 10 else \
                col2.number_input(f"V{i}", value=0.0, format="%.4f") if i <= 20 else \
                col3.number_input(f"V{i}", value=0.0, format="%.4f")
        input_values.append(value)

    amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0)
    time = st.number_input("⏱ Time (seconds since first transaction)", min_value=0, value=100000)

# ---- SCALE INPUTS ----
scaled_amount = scaler_amount.transform([[amount]])[0][0]
scaled_time = scaler_time.transform([[time]])[0][0]
final_features = np.array(input_values + [scaled_amount, scaled_time]).reshape(1, -1)

# ---- PREDICTION ----
if st.button("🔍 Detect Fraud"):
    pred = model.predict(final_features)[0]
    prob = model.predict_proba(final_features)[0][1]

    st.write("---")
    if pred == 1:
        st.error(f"🚨 **Fraudulent Transaction Detected!**")
        st.write(f"**Model Confidence:** {prob:.2%}")
        st.image("https://cdn-icons-png.flaticon.com/512/565/565547.png", width=120)
    else:
        st.success(f"✅ **Legitimate Transaction**")
        st.write(f"**Model Confidence:** {(1 - prob):.2%}")
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=120)

# ---- FOOTER ----
st.write("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Made by Vishakha Ahlawat | Powered by Streamlit</p>",
    unsafe_allow_html=True
)
