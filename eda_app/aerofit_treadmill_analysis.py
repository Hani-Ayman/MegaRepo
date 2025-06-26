#import libraries
import numpy as np
import pandas as pd
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt


st.set_page_config(page_title="Aerofit trademill analysis",layout="wide")
st.title("Aerofit Tradmill data dashboard")

uploaded_file=st.file_uploader("Please upload your dataset",type=['csv'])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Shape of the dataset")
    st.write(df.shape)

    st.subheader("Column names of my dataset")
    st.write(df.columns.to_list())

    #create few checkboxes
    data_type=st.radio("Choose options",("Data Info","Missing values","statistics"))
    if data_type=="Data Info":
        st.write("The datatypes in this dataset are:",df.info())
    elif data_type=="Missing_values":
        st.write("Missing values of the dataset:",df.isna().sum(axis=0))
    elif data_type=="statistics":
        st.write("dataset statistics are:",df.describe())

    numeric_cols =df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols= df.select_dtypes(include=["object"]).columns.tolist()
    st.write(numeric_cols)
    st.write(categorical_cols)

    #Count plot for numerical columns
    st.subheader("Count Plot")
    selected_cols= st.selectbox("select the numeric column:",numeric_cols)
    fig,ax = plt.subplots()
    sns.countplot(x=df[selected_cols],ax=ax)
    st.pyplot(fig)

    #Count plot for Catagorical columns
    st.subheader("Count Plot")
    cat_cols= st.selectbox("select the catagorical column:",categorical_cols)
    fig,ax = plt.subplots()
    sns.countplot(x=df[cat_cols],ax=ax)
    st.pyplot(fig)

    #Box plot
    st.subheader("Box plot")
    box_cols= st.selectbox("select the numeri column:",numeric_cols)
    fig,ax = plt.subplots()
    sns.boxplot(x=df[box_cols],ax=ax)
    st.pyplot(fig)

    #Hist plot
    st.subheader("Hist Plot")
    hist_cols= st.selectbox("select the numerical column:",numeric_cols)
    fig,ax = plt.subplots()
    sns.histplot(x=df[hist_cols],ax=ax)
    st.pyplot(fig)

    #Bi variate analysis
    st.subheader("Bi-Variate analysis of our data set")
    num_cols=st.selectbox("Select numeric column:",numeric_cols)
    catagory_cols=st.selectbox("Select catagorical col:",categorical_cols)
    fig,ax=plt.subplots()
    sns.boxplot(x=df[num_cols],y=df[catagory_cols])
    st.pyplot(fig)

    #scatterplot
    st.subheader("Bi-Variate analysis using scatterplot ")
    n_cols=st.selectbox("Select numerical column:",numeric_cols)
    catagor_cols=st.selectbox("Select catagoric col:",categorical_cols)
    fig,ax=plt.subplots()
    sns.scatterplot(x=df[n_cols],y=df[catagor_cols])
    st.pyplot(fig)

    #Multi-variate analysis
    #heatmap of dataset
    st.subheader("Co_relation heatmap")
    fig,ax=plt.subplots()
    sns.heatmap(df[numeric_cols].corr(),annot=True,cmap="magma",ax=ax)
    st.pyplot(fig)

    #Pairplot
    st.subheader("Pair plot of our dataset")
    fig =sns.pairplot(df[numeric_cols])
    st.pyplot(fig)

else:
    st.write("Please upload the dataset first for the exploratory data analysis")