import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("insurance_claim_pipeline.pkl")


st.set_page_config(
    page_title="Insurance Claim Prediction",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Health Insurance Claim Prediction")

st.write("Enter the details below to predict the insurance claim amount.")


with st.form("input_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0,max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=8, value=0)

    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", ["male", "female"])
        diabetic = st.selectbox("Diabetic", ["Yes", "No"])
        smoker = st.selectbox("Smoker", ["Yes", "No"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict Claim")


if submitted:

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction_log = model.predict(input_data)
    prediction = np.expm1(prediction_log)

    st.success(f"Estimated Insurance Payment Amount: ${prediction[0]:,.2f}")

    st.markdown("---")
