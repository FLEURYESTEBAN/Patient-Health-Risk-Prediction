# Use an official Python runtime as the base image
FROM python:3.9-slim


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY Final_Machine_Learning_Diabetes /app/

# Specify the command to run the script
CMD ["python", "main.py"]
