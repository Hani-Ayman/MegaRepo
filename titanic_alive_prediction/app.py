import streamlit as st
import pandas as pd
import joblib

# Load models and encoder

model= joblib.load("Logistic_regression.pkl")
le = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival using multiple models.")

# Input form
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0.42, 80.0, 30.0)
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)

# Prepare input
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "Sex": [le.transform([sex])[0]],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

if st.button("Predict Survival"):
    st.subheader("Model Predictions:")
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    result = "Survived" if pred == 1 else "Did not survive"
    st.write(f"{result}", f"({prob:.2%} survival probability)" if prob is not None else "")