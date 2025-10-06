# Use an official Python base image
FROM python:3.13-slim-bullseye

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements (if you have one)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code and data/logs directories
COPY src/ ./src/
COPY data/ ./data/
COPY logs/ ./logs/

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application
CMD ["python", "-m", "src"]