#Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

#Uploading dataset to the dataframe
df=pd.read_csv('daily_minimum_temps.csv',parse_dates=["Date"],index_col="Date")

#df.head()
#Finding out missing values
#df.isna().sum(axis=0)
#Checking if any value in the dat   aset is string format
df["Temp"]=pd.to_numeric(df["Temp"],errors="coerce")#Coerce make unknown values to NaN
#df.isna().sum(axis=0)
df=df.dropna()

#Normalize the Features
scaler=MinMaxScaler()
data_scaled=scaler.fit_transform(df["Temp"].values.reshape(-1,1))

#Sequence length for temperature data
seq_length=30

#Create a function for creating sequences
def create_sequences(data_scaled,seq_length):
    X,y=[],[]
    for i in range(len(data_scaled)-seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    return np.array(X),np.array(y)

#Calling the function and storing the values in X and y
X,y=create_sequences(data_scaled,seq_length)

#Divide the dataset into train test and split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

#Building model
model=Sequential([
    LSTM(64,activation="relu",input_shape=(seq_length,1)),
    Dense(1)#Output layer. Since output is a single value
])

model.compile(optimizer="adam",loss=MeanSquaredError())

model.fit(X_train,y_train,epochs=20,batch_size=32)

#Make predictions
y_pred_scaled=model.predict(X_test)

#Inverse Transform the scaled data
y_pred_scaled=np.clip(y_pred_scaled,0,1)
y_pred=scaler.inverse_transform(y_pred_scaled)
y_test_actual=scaler.inverse_transform(y_test)

#Predict the next day temperature
last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled=np.clip(next_temp_scaled,0,1)
next_day_temp = scaler.inverse_transform(next_temp_scaled)
print("Predicted next day temperature is:",next_day_temp)

# Save model and scaler
model.save("lstm_temp_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")