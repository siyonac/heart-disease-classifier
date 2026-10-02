import streamlit as st
import pandas as pd
import joblib


# Load the trained model
model = joblib.load("model.pkl")


st.title("Heart Disease Classifier")
st.write(
    "Enter patient information below to see the model's classification."
)

st.warning(
    "This project is for educational purposes only and is not a medical diagnostic tool."
)


st.header("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1]
)

restecg = st.selectbox(
    "Resting ECG Results",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise-Induced Angina",
    options=[0, 1]
)

oldpeak = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    options=[3, 6, 7]
)


if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")
    st.write(f"Model probability of heart disease: {probability:.1%}")

    if prediction == 1:
        st.error("The model classified this input as: Heart disease present")
    else:
        st.success("The model classified this input as: No significant heart disease")