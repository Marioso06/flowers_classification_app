<<<<<<< HEAD
# Base Image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch torchvision torchaudio

# Copy all files except those in dockerignore
COPY . .
RUN mkdir -p data/raw data/processed data/external


ENV PYTHONPATH=/app
ENV FLASK_APP=predict_api.python

EXPOSE 9000

CMD [ "python", 'predict_api.py' ]
=======
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
ENV FLASK_APP=predict_api.py

# Expose port for the API
EXPOSE 9000

# Default command runs the API
CMD ["python", "predict_api.py"]
>>>>>>> refs/remotes/origin/main
