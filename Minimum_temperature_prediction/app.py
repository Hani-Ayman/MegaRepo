import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("lstm_temp_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Min Temperature Predictor")
st.title("Minimum Temperature Forecast")

#Uploading dataset to the dataframe
df = pd.read_csv("daily_minimum_temps.csv", parse_dates=["Date"], index_col="Date")
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
df.dropna(inplace=True)
data_scaled = scaler.transform(df["Temp"].values.reshape(-1, 1))

#Sequence length for temperature data
seq_length = 30
last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)

# Predict next day's temperature
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
next_day_temp = scaler.inverse_transform(next_temp_scaled)[0][0]

st.subheader("Predicted Minimum Temperature for Tomorrow:")
st.success(f"{next_day_temp:.2f} °C")

# Optional: Show recent temperature trend
if st.checkbox("Show last 30 days temperature trend"):
    st.line_chart(df["Temp"].tail(30))