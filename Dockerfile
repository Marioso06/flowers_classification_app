FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for container simplicity)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models predictions data/raw data/processed data/external

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api_main.py

# Expose port for the API
EXPOSE 9000

# Default command runs the API
CMD ["python", "api_main.py"]
