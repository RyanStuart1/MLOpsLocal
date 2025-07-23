import streamlit as st
import joblib
import numpy as np
import os
from pipeline.chatbot import show_chatbot_sidebar
from pages_components.home_button import render_home_button

# Load trained model
model_path = "models/model.pkl"
if not os.path.exists(model_path):
    st.error("model.pkl not found. Please run your training pipeline first.")
    st.stop()

model = joblib.load(model_path)

st.set_page_config(layout="centered")
render_home_button()
st.title("Credit Risk Prediction")

st.markdown("Fill in the applicant's loan and financial info to assess default risk.")

# Input fields with safe, bounded values
age = st.number_input("Age", min_value=18, max_value=100, value=30)
credit_score = st.slider("Credit Score", min_value=300, max_value=999, value=650)
loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000)
loan_term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
employment_length = st.slider("Employment Length (years)", min_value=0, max_value=40, value=5)
annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=300000, value=60000)
dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
open_accounts = st.slider("Number of Open Accounts", min_value=0, max_value=20, value=5)
delinquencies = st.slider("Past Delinquencies", min_value=0, max_value=10, value=0)

# Combine into input array for model
input_data = np.array([[age, credit_score, loan_amount, loan_term_months,
                        employment_length, annual_income, dti,
                        open_accounts, delinquencies]])

if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    st.success("✅ Low Risk" if prediction == 0 else "⚠️ High Risk: Likely Default")

show_chatbot_sidebar()