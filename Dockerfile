# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app will run on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
