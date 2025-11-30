import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.title("🩺 Healthcare Predictive Analytics App")
st.subheader("Diabetes Risk Prediction using Machine Learning")

st.markdown("Enter patient details below to get diabetes risk prediction.")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bloodpressure = st.number_input("Blood Pressure", 0, 200, 70)
skinthickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    # Create input dataframe
    input_data = pd.DataFrame({
        "pregnancies": [pregnancies],
        "glucose": [glucose],
        "bloodpressure": [bloodpressure],
        "skinthickness": [skinthickness],
        "insulin": [insulin],
        "bmi": [bmi],
        "diabetespedigreefunction": [dpf],
        "age": [age]
    })

    # Predict probability
    prediction = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Diabetes Risk Score: {prediction:.2f}**")

    if prediction >= 0.5:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✔ Low Risk of Diabetes")

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning + Streamlit")
