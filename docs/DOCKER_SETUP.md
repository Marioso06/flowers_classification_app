# Flowers Classification App - Docker Setup Guide

This guide explains how to use Docker to containerize the Flowers Classification Application, including both the prediction API and MLflow tracking server.

## Overview

The containerization consists of:

1. **Main Application Container**: Runs the Flask-based prediction API
2. **MLflow Container**: Provides experiment tracking and artifact storage

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- Basic understanding of Docker concepts

## Setup and Usage

### 1. Building and Starting the Containers

```bash
# Start both the API and MLflow containers
docker-compose up -d

# To see logs
docker-compose logs -f
```

### 2. Stopping the Containers

```bash
# Stop all containers
docker-compose down
```

## Container Structure

### Main Application Container

- Based on Python 3.10 with essential libraries
- Exposes port 9000 for the prediction API
- Mounts volumes for models, data, and predictions
- Environment variables:
  - `MLFLOW_TRACKING_URI`: URL to MLflow server
  - `FLASK_HOST`: Host to bind the Flask app (default: 0.0.0.0)
  - `FLASK_PORT`: Port to bind the Flask app (default: 9000)
  - `FLASK_DEBUG`: Enable/disable debug mode (default: False)

### MLflow Container

- Dedicated container for experiment tracking
- Exposes port 5000 for the MLflow UI and API
- Persists data through a named volume
- Configured to use SQLite for backend storage and local file system for artifacts

## MLflow Integration: Logs and Artifacts

### How MLflow Logging Works Across Containers

1. **Communication Model**:
   - The main application connects to MLflow server using the `MLFLOW_TRACKING_URI` environment variable
   - This URI is set to `http://mlflow:5000` in the docker-compose.yml

2. **Logging Process**:
   - When `mlflow.log_metric()`, `mlflow.log_param()`, or other tracking calls are made:
     - The Python MLflow client in the main container sends HTTP requests to the MLflow server
     - The server processes these requests and stores them in its database

3. **Artifact Storage**:
   - When artifacts (models, plots, files) are logged via `mlflow.log_artifact()`:
     - Files are sent from the main application to the MLflow server over HTTP
     - The MLflow server stores these in its artifact directory
   - With our Docker setup, artifacts are stored in the persistent volume `mlflow-data`

4. **Persistence**:
   - MLflow data persists even if containers are restarted
   - The docker volume `mlflow-data` ensures data isn't lost when containers are removed

### Benefits of Separate MLflow Container

1. **Scalability**: MLflow can be scaled independently
2. **Isolation**: Clean separation of concerns
3. **Persistence**: Experiment data is preserved if application container is replaced
4. **Sharing**: Multiple applications can use the same MLflow server

## Interacting with the Prediction API

### 1. Using the API Directly

The prediction API is available at `http://localhost:9000` with these endpoints:

- `GET /flowers_classification_home`: Information about the API
- `GET /helth_status`: Health check
- `POST /v1/predict`: Prediction using model v1
- `POST /v2/predict`: Prediction using model v2

Example request using curl:

```bash
curl -X POST http://localhost:9000/v1/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

Where `request.json` contains:

```json
{
  "image_data": "base64_encoded_image_string",
  "top_k": 5
}
```

### 2. Creating a Client Application

For users who prefer a GUI, a simple client application can be created. Example approaches:

1. **Web-based Client**:
   - Create a simple HTML/JS page that uploads images and displays results
   - Example using fetch API:

```javascript
async function predict(imageBase64) {
  const response = await fetch('http://localhost:9000/v1/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      image_data: imageBase64,
      top_k: 5
    })
  });
  return response.json();
}
```

2. **Python Client**:
   - Using the requests library:

```python
import requests
import base64

def predict_flower(image_path, top_k=5):
    # Read image and convert to base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Make API request
    response = requests.post(
        "http://localhost:9000/v1/predict",
        json={"image_data": encoded_string, "top_k": top_k}
    )
    
    return response.json()
```

## Training in the Container

To run training in the container:

```bash
docker-compose run app python src/train.py --data_directory /app/data
```

This will automatically log metrics and parameters to the MLflow server running in the other container.

## Advanced Configuration

### Customizing MLflow Storage

For production, you might want to use external databases and storage:

```yaml
# Example MySQL and S3 configuration in docker-compose.yml
mlflow:
  environment:
    - MLFLOW_TRACKING_URI=mysql+pymysql://user:password@mysql-server/mlflow
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    - AWS_ACCESS_KEY_ID=minioadmin
    - AWS_SECRET_ACCESS_KEY=minioadmin
  command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mysql+pymysql://user:password@mysql-server/mlflow --default-artifact-root s3://mlflow
```

### GPU Support

To use GPU with Docker:

1. Install NVIDIA Container Toolkit
2. Modify Dockerfile to use CUDA base image
3. Add GPU runtime to docker-compose.yml:

```yaml
app:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Troubleshooting

1. **Model Loading Issues**:
   - Check that model files exist in the mounted volume
   - Verify paths in environment variables

2. **MLflow Connection Issues**:
   - Ensure MLflow container is running
   - Check network connectivity between containers
   - Review logs using `docker-compose logs mlflow`

3. **API Not Responding**:
   - Verify the container is running: `docker-compose ps`
   - Check logs for errors: `docker-compose logs app`
