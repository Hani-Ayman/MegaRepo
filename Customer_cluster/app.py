import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st

kmeans=joblib.load("Model.pkl")
df = pd.read_csv("Mall_Customers.csv")

#df.info()
#print(df.shape)

X = df[["Annual Income (k$)","Spending Score (1-100)"]]

st.set_page_config(page_title="",layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the customers annual income and spending score to predict the cluster.")

income=st.number_input("Annual Income of a customer",min_value=0,max_value=400,value=50)
spending=st.slider("Spending Score between 1 to 100",1,100,20)

if st.button("Predict Cluster"):
    input_data=np.array([[income,spending]])
    cluster=kmeans.predict(input_data)
    st.success(f"Predicted cluster is :{cluster}")