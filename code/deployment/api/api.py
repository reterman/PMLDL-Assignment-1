from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Path to the saved models
model_path = "/app/models/linear_regression_model.pkl"

# Load the model and scaler from the specified file
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Description of the data that will be received in requests
class DiabetesData(BaseModel):
    age: float  # Age of the patient
    sex: float  # Sex of the patient (0 for female, 1 for male)
    bmi: float  # Body Mass Index
    bp: float   # Blood Pressure
    s1: float   # Blood test feature 1
    s2: float   # Blood test feature 2
    s3: float   # Blood test feature 3
    s4: float   # Blood test feature 4
    s5: float   # Blood test feature 5
    s6: float   # Blood test feature 6

# Route for predictions
@app.post("/predict/")
def predict(data: DiabetesData):
    # Convert the input data into a NumPy array
    input_data = np.array([[data.age, data.sex, data.bmi, data.bp, data.s1, data.s2, data.s3, data.s4, data.s5, data.s6]])
    
    # Get the prediction from the model
    prediction = model.predict(input_data)
    
    # Return the result as a JSON response
    return {"prediction": prediction[0]}
