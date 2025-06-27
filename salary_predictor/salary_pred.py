import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib #to create and save the model


df=pd.read_csv("salary_data.csv")
#print(df.info())

#split data in target var and independent var
X=df[["YearsExperience"]]
y=df[["Salary"]]

#train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1) #20% data for test data

#model selection
#creating an object of StandardScaler module present in sklearn library
scaler=StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

#train
model= LinearRegression()
model.fit(X_train_scaled,y_train)

#save the model and scaler
joblib.dump(model,"salary_pred.pkl")
joblib.dump(scaler,"scaler.pkl")
print("model and scaler saved succesfully")

