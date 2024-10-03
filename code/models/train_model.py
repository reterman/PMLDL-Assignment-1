import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import pickle
import os

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model using Mean Squared Error (MSE)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model to a file
model_dir = "models"  # Directory to save the model
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
model_path = os.path.join(model_dir, "linear_regression_model.pkl")  # Define the model file path

with open(model_path, "wb") as f:
    pickle.dump(model, f)  # Save the model using pickle

print(f"Model saved to {model_path}")  # Print confirmation of the saved model
