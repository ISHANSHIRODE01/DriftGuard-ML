# 1. Use an official, lightweight Python base image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the model and data directories (Production artifacts)
COPY model/ ./model/
COPY data/ ./data/

# 7. Copy the application code and necessary modules
COPY app.py .
COPY drift_detection.py .
COPY model_versioning.py .

# 8. Expose the port FastAPI will run on
EXPOSE 8000

# 9. Define the command to run the application
# We use 4 workers for basic production concurrency
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
