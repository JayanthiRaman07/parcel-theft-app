import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Parcel Theft Predictor", page_icon="📦")

# Title
st.title("📦 Parcel Theft Risk Predictor")
st.markdown("Predict the risk of parcel theft based on delivery conditions.")

# Load model
model = joblib.load("parcel_model.pkl")

# Sidebar inputs
st.sidebar.header("Enter Delivery Details")

time = st.sidebar.selectbox("Delivery Time", ["Morning", "Afternoon", "Evening"])
location = st.sidebar.selectbox("Location Type", ["Apartment", "House"])
person = st.sidebar.selectbox("Person at Home", ["No", "Yes"])
cctv = st.sidebar.selectbox("CCTV Available", ["No", "Yes"])
delivery = st.sidebar.selectbox("Delivery Type", ["Doorstep", "Handed"])

# Mapping
time_map = {"Morning":0, "Afternoon":1, "Evening":2}
location_map = {"Apartment":0, "House":1}
person_map = {"No":0, "Yes":1}
cctv_map = {"No":0, "Yes":1}
delivery_map = {"Doorstep":0, "Handed":1}

# Prediction
if st.button("Predict Risk"):
    sample = np.array([[time_map[time], location_map[location], person_map[person], cctv_map[cctv], delivery_map[delivery]]])
    
    prob = model.predict_proba(sample)[0][1]

    # Result
    st.subheader("Prediction Result")
    st.write(f"Risk Score: {round(prob,2)}")

    if prob < 0.3:
        st.success("Low Risk 🟢")
    elif prob < 0.7:
        st.warning("Medium Risk 🟡")
    else:
        st.error("High Risk 🔴")

    st.progress(float(prob))

    # Explanation
    st.subheader("Why this risk?")

    if cctv == "No":
        st.write("• No CCTV increases risk")

    if person == "No":
        st.write("• No one at home increases risk")

    if time == "Evening":
        st.write("• Evening deliveries are riskier")

    # Recommendation
    st.subheader("Recommendation")

    if prob > 0.7:
        st.write("• Use CCTV or secure locker")
        st.write("• Ensure someone is home")
        st.write("• Prefer handed delivery")
    else:
        st.write("• Delivery conditions are relatively safe")

    # Graph
    st.subheader("Risk Insight")

    data = pd.DataFrame({
        "Condition": ["Safe", "Risky"],
        "Level": [1-prob, prob]
    })

    st.bar_chart(data.set_index("Condition"))

# Footer
st.markdown("---")
st.caption("Developed using Streamlit for Parcel Theft Prediction")
