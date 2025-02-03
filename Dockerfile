FROM ubuntu:22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies for the application
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application and model weights
COPY app /app
WORKDIR /app

# Expose port
EXPOSE 8005

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
