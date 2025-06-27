#Importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score, confusion_matrix, classification_report
import joblib
import streamlit as st

model= tf.keras.models.load_model("Diabetes_neural_network/Dabetic_model.h5")
scaler= joblib.load("Diabetes_neural_network/scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction")
st.title("Diabetes Prediction")
st.markdown("Enter the following details")

#input feilds
pregnancy= st.number_input("Number of times pregnant: ",min_value=0)
glucose= st.number_input("Glucose: ",min_value=0)
blood_pressure= st.number_input("Blood pressure: ",min_value=0)
skin_thickness= st.number_input(" skin  thickness: ",min_value=0)
insulin= st.number_input("insulin in bofy: ",min_value=0)
bmi= st.number_input("Body mass index: ",min_value=1)
diabetes_pedigree_function= st.number_input("Enter Diabetes pedigree function: ",min_value=0)
age= st.number_input("Enter Age: ",min_value=0)

#Make predictions 
if st.button("Predict Diabetes"):
    input_data= np.array([[pregnancy,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]])
    input_scaled=scaler.transform(input_data)
    prediction= model.predict(input_scaled)[0][0]
    result= "Not Diabetic" if prediction<0.5 else "Diabetic"
    st.write("According to the input data we think you are: ",result)
