# Use the official Python 3.9 slim image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the API script into the container's working directory
COPY api.py /app/

# Install the necessary dependencies
RUN pip install fastapi uvicorn numpy scikit-learn pydantic

# Command to run the application using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
