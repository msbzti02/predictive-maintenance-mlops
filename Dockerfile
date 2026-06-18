FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries (LightGBM, etc.)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire MLOps repository
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server to serve the MLflow Production Model
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
