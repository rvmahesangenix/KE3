# Dockerfile

FROM ubuntu:22.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the app and model weights
COPY app /app
WORKDIR /app

# Expose port for FastAPI
EXPOSE 8005

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
