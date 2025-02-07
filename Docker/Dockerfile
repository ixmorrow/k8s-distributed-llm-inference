FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into the container
COPY src/ /app/

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]