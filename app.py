import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from data_quality import CLASS_NAMES, TARGET_COLUMN, load_clean_data

st.set_page_config(page_title="STEM Student Retention", layout="wide")
st.title("STEM Student Success & Dropout Predictor")

@st.cache_data
def get_data():
    return load_clean_data()


df, audit_details = get_data()
page = st.sidebar.selectbox("Navigate", ["Data Overview", "Data Audit", "Risk Predictor"])
    
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
        ax2.set_xticklabels(["Not a Debtor", "Debtor"])
        st.pyplot(fig2)

elif page == "Data Audit":
    st.header("Data Auditing, Cleaning, and Class Balance")

    raw_audit = audit_details["raw_audit"]
    cleaned_audit = audit_details["cleaned_audit"]
    balance = cleaned_audit["class_balance"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", cleaned_audit["rows"])
    col2.metric("Columns", cleaned_audit["columns"])
    col3.metric("Duplicate Rows", cleaned_audit["duplicate_rows"])
    col4.metric("Missing Cells", cleaned_audit["missing_cells"])

    st.subheader("Cleaning Actions")
    for action in audit_details["cleaning_actions"]:
        st.write(f"- {action}")

    st.subheader("Audit Checks")
    checks = pd.DataFrame(
        [
            {
                "Check": "Dirty column names in raw file",
                "Result": len(raw_audit["dirty_column_names"]),
                "Details": ", ".join(raw_audit["dirty_column_names"]) or "None",
            },
            {
                "Check": "Non-numeric feature columns after cleaning",
                "Result": len(cleaned_audit["non_numeric_features"]),
                "Details": ", ".join(cleaned_audit["non_numeric_features"]) or "None",
            },
            {
                "Check": "Constant feature columns",
                "Result": len(cleaned_audit["constant_features"]),
                "Details": ", ".join(cleaned_audit["constant_features"]) or "None",
            },
            {
                "Check": "Unexpected target labels",
                "Result": len(cleaned_audit["unexpected_targets"]),
                "Details": ", ".join(cleaned_audit["unexpected_targets"]) or "None",
            },
        ]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)

    st.subheader("Class Imbalance")
    balance_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Count": [int(balance["counts"][class_name]) for class_name in CLASS_NAMES],
            "Percent": [float(balance["percentages"][class_name]) for class_name in CLASS_NAMES],
        }
    )
    st.dataframe(balance_df, use_container_width=True, hide_index=True)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=balance_df, x="Class", y="Count", ax=ax3, palette="Set2")
    ax3.set_title("Target Class Distribution")
    ax3.set_xlabel("Student Outcome")
    ax3.set_ylabel("Number of Students")
    st.pyplot(fig3)

    st.write(f"Majority class: **{balance['majority_class']}**")
    st.write(f"Minority class: **{balance['minority_class']}**")
    st.write(f"Majority/minority ratio: **{balance['majority_minority_ratio']}:1**")
    if balance["is_imbalanced"]:
        st.warning("Class imbalance detected: the minority class is under 20% or the ratio is at least 2:1.")
    else:
        st.success("No class imbalance warning triggered.")
        
elif page == "Risk Predictor":
    st.header("Logistic Regression Risk Predictor")
        
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    st.write("Input student baseline metrics to calculate their predicted status:")
    student_index = st.number_input("Select a sample Student ID from dataset to test:", min_value=0, max_value=len(df)-1, value=0)
    sample_student = df.drop(columns=[TARGET_COLUMN]).iloc[student_index].copy()
            
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
