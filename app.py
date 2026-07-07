import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import streamlit as st

st.title("🎓 Student Dropout & Success Predictor")
st.write("Exploratory Data Analysis Dashboard")

@st.cache_data
def load_data():
    data = pd.read_csv('data/data.csv', sep=';')
    return data

try:
    df = load_data()
    
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Students", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    
    if st.checkbox("Show Raw Data Preview"):
        st.write(df.head(10))
        
    st.subheader("Distribution of Student Status")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Target', ax=ax, palette='Set2')
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Please ensure 'student_data.csv' is placed inside the 'data/' directory.")