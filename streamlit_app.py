import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")

st.write("Simple AI-based heart risk predictor.")

age = st.number_input("Age", 1, 100)
chol = st.number_input("Cholesterol", 100, 400)
bp = st.number_input("Blood Pressure", 80, 200)
hr = st.number_input("Heart Rate", 50, 220)

if st.button("Predict"):

    risk = 0

    if age > 50:
        risk += 25

    if chol > 220:
        risk += 25

    if bp > 140:
        risk += 25

    if hr < 120:
        risk += 25

    st.subheader(f"Risk Score: {risk}%")

    if risk < 40:
        st.success("Low Risk")
    elif risk < 70:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

st.markdown("---")
st.caption("Educational purpose only")