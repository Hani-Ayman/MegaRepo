import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
import re
import joblib
import streamlit as st

nltk.download("stopwords")
stop_words=set(stopwords.words('english'))

#Load model and vectorizer
model=joblib.load("Sentiment_analysis/sentiment_analysis_model.pkl")
vectorizer=joblib.load("Sentiment_analysis/vectorizer.pkl")

#Clean the text
def clean_text(text):
  text= re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens=text.split()
  tokens=[word for word in tokens if word not in stop_words]
  return " ".join(tokens)

#Designing the layout
st.set_page_config(page_title="Sentiment Analysis")
st.title("Sentiment Analysis of Movie reviews")
st.write("Write a movie review to predict its sentiment")

user_input=st.text_area("Write a movie review: ")
if st.button("Predict sentiment:"):
    cleaned=clean_text(user_input)
    transformed_input=vectorizer.transform([cleaned])
    prediction=model.predict(transformed_input)[0]
    sentiment= "Positive" if prediction ==1 else "Negative"
    st.success("The sentiment of the given review is:"+sentiment)