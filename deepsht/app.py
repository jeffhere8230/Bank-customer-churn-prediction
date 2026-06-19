import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# =========================
# Load Model and Scaler
# =========================

model = tf.keras.models.load_model("churn_model.keras")
scaler = joblib.load("scaler.pkl")

# =========================
# App Title
# =========================

st.set_page_config(page_title="Bank Churn Prediction")

st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a customer will leave the bank.")

# =========================
# User Inputs
# =========================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# =========================
# Encode Inputs
# =========================

geo_mapping = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

gender_mapping = {
    "Female": 0,
    "Male": 1
}

geo = geo_mapping[geography]
gen = gender_mapping[gender]

# =========================
# Prediction Button
# =========================

if st.button("Predict Churn"):

    data = pd.DataFrame(
        [[
            credit_score,
            geo,
            gen,
            age,
            tenure,
            balance,
            num_products,
            has_cr_card,
            is_active_member,
            estimated_salary
        ]],
        columns=[
            'CreditScore',
            'Geography',
            'Gender',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary'
        ]
    )

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    probability = prediction[0][0]

    st.subheader("Prediction Result")
    st.write(f"Probability of Churn: **{probability:.2%}**")

    if probability > 0.5:
        st.error("⚠️ Customer is likely to leave the bank.")
    else:
        st.success("✅ Customer is likely to stay with the bank.")