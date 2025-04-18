FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --verbose

# Install PyTorch based on architecture
ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      pip install torch torchvision torchaudio; \
    else \
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

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
