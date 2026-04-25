import streamlit as st
import joblib
import numpy as np

st.title("Parcel Theft Risk Predictor")

# Load model
try:
    model = joblib.load("parcel_model.pkl")
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Inputs
time = st.selectbox("Delivery Time", ["Morning", "Afternoon", "Evening"])
location = st.selectbox("Location Type", ["Apartment", "House"])
person = st.selectbox("Person at Home", ["No", "Yes"])
cctv = st.selectbox("CCTV Available", ["No", "Yes"])
delivery = st.selectbox("Delivery Type", ["Doorstep", "Handed"])

# Convert to numbers
time_map = {"Morning":0, "Afternoon":1, "Evening":2}
location_map = {"Apartment":0, "House":1}
person_map = {"No":0, "Yes":1}
cctv_map = {"No":0, "Yes":1}
delivery_map = {"Doorstep":0, "Handed":1}

if st.button("Predict Risk"):
    sample = np.array([[time_map[time], location_map[location], person_map[person], cctv_map[cctv], delivery_map[delivery]]])
    prob = model.predict_proba(sample)[0][1]
    st.write(f"Risk Score: {round(prob,2)}")
    if prob < 0.3:
    st.success("Low Risk 🟢")
elif prob < 0.7:
    st.warning("Medium Risk 🟡")
else:
    st.error("High Risk 🔴")
