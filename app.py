import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="STEM Student Retention", layout="wide")
st.title("STEM Student Success & Dropout Predictor")

@st.cache_data
def get_data():
    return pd.read_csv('data/data.csv', sep=';')


df = get_data()
page = st.sidebar.selectbox("Navigate", ["Data Overview", "Risk Predictor"])
    
if page == "Data Overview":
        st.header("Exploratory Data Analysis (EDA)")
        st.write("Let's analyze the relationships between student demographics, grades, and dropout rates.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students Tracked", df.shape[0])
        col2.metric("Average Age at Enrollment", f"{df['Age at enrollment'].mean():.1f} years")
        scholarship_pct = (df['Scholarship holder'].sum() / len(df)) * 100
        col3.metric("Scholarship Holders", f"{scholarship_pct:.1f}%")
        
        st.markdown("---")
        
        st.subheader("Academic Performance & Student Success")
        st.write("How do 1st-semester grades (0-20 scale) correlate with whether a student graduates or drops out?")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (grade)', ax=ax, palette='Set2')
        ax.set_title("1st Semester Grades Distribution by Student Outcome")
        ax.set_xlabel("Student Outcome")
        ax.set_ylabel("1st Sem Grade (0-20)")
        st.pyplot(fig)
        
        st.subheader("Financial Demographics")
        st.write("Let's look at how being a debtor (owing tuition) impacts dropout rates.")
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='Debtor', hue='Target', ax=ax2, palette='Set1')
        ax2.set_title("Dropout Rates: Debtors vs Non-Debtors ")
        ax2.set_xlabel("Is the Student a Debtor?")
        ax2.set_xticklabels(["Debtor", "Not a Debtor"])
        st.pyplot(fig2)
        
elif page == "Risk Predictor":
    st.header("Logistic Regression Risk Predictor")
        
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    st.write("Input student baseline metrics to calculate their predicted status:")
    student_index = st.number_input("Select a sample Student ID from dataset to test:", min_value=0, max_value=len(df)-1, value=0)
    sample_student = df.drop(columns=['Target']).iloc[student_index].copy()
            
    st.markdown("### Tweak Key Metrics for this Student:")
    gpa_1st_sem = st.slider("1st Semester GPA (0-20 scale)", 0.0, 20.0, float(sample_student['Curricular units 1st sem (grade)']))
    age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=int(sample_student['Age at enrollment']))
            
    sample_student['Curricular units 1st sem (grade)'] = gpa_1st_sem
    sample_student['Age at enrollment'] = age
            
    if st.button("Calculate Probability Profile"):
        features_vector = np.array(sample_student).reshape(1, -1)
        scaled_vector = scaler.transform(features_vector)
                
        prediction_class = model.predict(scaled_vector)[0]
        probabilities = model.predict_proba(scaled_vector)[0]
                
        status_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
                
        st.markdown("---")
        st.subheader(f"Predicted Outcome: **{status_mapping[prediction_class]}**")
        st.write(f"Probability of Dropping Out: **{probabilities[0]*100:.2f}%**")
        st.write(f"Probability of Staying Enrolled: **{probabilities[1]*100:.2f}%**")
        st.write(f"Probability of Graduating: **{probabilities[2]*100:.2f}%**")