import streamlit as st
import requests

# Title of the application
st.title("Diabetes Prediction App")

# Instructions for the user to enter model features
st.write("Enter the feature values for prediction:")

# Input fields for the model features
age = st.number_input("Age", min_value=0, max_value=120, value=35)
sex = st.number_input("Sex (0 for female, 1 for male)", min_value=0, max_value=1, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=30.0)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
s1 = st.number_input("S1", min_value=-100.0, max_value=100.0, value=0.0)
s2 = st.number_input("S2", min_value=-100.0, max_value=100.0, value=0.0)
s3 = st.number_input("S3", min_value=-100.0, max_value=100.0, value=0.0)
s4 = st.number_input("S4", min_value=-100.0, max_value=100.0, value=0.0)
s5 = st.number_input("S5", min_value=-100.0, max_value=100.0, value=0.0)
s6 = st.number_input("S6", min_value=-100.0, max_value=100.0, value=0.0)

# Button to submit data for prediction
if st.button("Make Prediction"):
    # Prepare the input data for the API request
    input_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "bp": bp,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "s6": s6,
    }
    
    # Send a POST request to the FastAPI (assumed to be running locally on port 8000)
    response = requests.post("http://api:8000/predict/", json=input_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Retrieve the prediction result from the response
        result = response.json().get("prediction")
        st.success(f"Predicted outcome: {result} (This value represents a quantitative measure of disease progression after one year from the baseline)")
    else:
        st.error("Error in API request")
