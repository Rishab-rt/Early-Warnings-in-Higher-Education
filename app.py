import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="STEM Student Retention", layout="wide")
st.title("🎓 STEM Student Success & Dropout Predictor")

@st.cache_data
def get_data():
    return pd.read_csv('data/data.csv', sep=';')

try:
    df = get_data()
    page = st.sidebar.selectbox("Navigate", ["Data Overview", "Risk Predictor"])
    
    if page == "Data Overview":
        st.header("📊 Dataset Insights")
        st.write(f"The dataset currently contains **{df.shape[0]}** student records across **{df.shape[1]}** metrics.")
        st.write("Use the sidebar to switch to the live prediction tool.")
        st.dataframe(df.head(10))
        
    elif page == "Risk Predictor":
        st.header("🔮 Logistic Regression Risk Predictor")
        
        try:
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
                st.write(f"📉 Probability of Dropping Out: **{probabilities[0]*100:.2f}%**")
                st.write(f"⏳ Probability of Staying Enrolled: **{probabilities[1]*100:.2f}%**")
                st.write(f"🎓 Probability of Graduating: **{probabilities[2]*100:.2f}%**")
                
        except FileNotFoundError:
            st.warning("⚠️ Model files not found. Run `python model.py` in your terminal first to train and save the model.")

except FileNotFoundError:
    st.error("Data file missing. Please ensure 'data/student_data.csv' exists.")