# Use the official Python image as a base
FROM python:3.8-slim-buster

# Set environment variables
RUN apt update -y && apt install awscli -y

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Run the application
CMD ["python3", "app.py"]
