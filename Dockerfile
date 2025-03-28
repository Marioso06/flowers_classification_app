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
