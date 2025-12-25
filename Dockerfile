# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# (None currently needed for this app, but good practice to have apt-get update)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY backend/requirements.txt /app/backend/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . /app

# Expose port 5001
EXPOSE 5001

# Define environment variable
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "5001"]
