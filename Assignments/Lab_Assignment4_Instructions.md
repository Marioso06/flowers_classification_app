# Lab Assignment 4: Containerizing Your Machine Learning Project

## Overview

In this lab assignment, you will containerize your machine learning project using Docker. Containerization offers several advantages for ML deployments:

* **Reproducibility**: Ensures consistent environment across development, testing, and production.
* **Portability**: Allows your application to run consistently on any system that supports Docker.
* **Isolation**: Keeps dependencies and configurations separate from the host system.
* **Scalability**: Facilitates easier scaling of your application in production environments.
* **Versioning**: Enables versioning of your entire application environment, not just code.

You will create two Docker containers:
1. A container for your machine learning application that includes your training, prediction, and API code.
2. A container for MLflow to track experiments and manage model lifecycle.

Both containers will communicate via an internal Docker network.

## Docker Fundamentals

Before diving into the assignment, let's review some key Docker concepts:

### Docker Images and Containers

* **Docker Image**: A lightweight, standalone, executable package that includes everything needed to run an application: code, runtime, libraries, environment variables, and configuration files.
* **Container**: A running instance of a Docker image. Containers are isolated from each other and from the host machine.

### Dockerfile

A Dockerfile is a text document containing commands to assemble a Docker image. Here's a simple example:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

### Multiple Dockerfiles

When you have multiple services that require different configurations, it's common to create multiple Dockerfiles with extensions to differentiate them:

```
Dockerfile.mlapp    # For your ML application
Dockerfile.mlflow   # For MLflow
```

This approach helps organize your containerization files when dealing with multiple services.

### Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application's services, networks, and volumes. 

Key benefits of Docker Compose:
* Manages multiple containers as a single application
* Defines networks for inter-container communication
* Sets up volumes for persistent data
* Manages environment variables
* Orchestrates container startup order

## Assignment Instructions

### 1. Create a Dockerfile for Your ML Application

Create a file named `Dockerfile.mlapp` in your project root with the following structure (see the example how it is done in the Flowers Classification App repo):

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files for your application
# Include your model files, source code, and configuration files or any file you need in your container
COPY 

# Expose the port your Flask API will run on
EXPOSE 5000

# Command to run your prediction API
CMD 
```

**Key Considerations for Your ML Application Container:**

1. **File Selection**: Carefully select which files to copy into your container. You need:
   * Source code files (`src/train.py`, `src/predict.py`, `src/predict_api.py`, etc.)
   * Model files (pre-trained models in `models/`)
   * Configuration files (`configs/`)
   
2. **Data Management**: For your datasets, consider one of these approaches:
   * **Option 1**: Copy the data folder into the container
     ```dockerfile
     COPY data/ /app/data/
     ```
   * **Option 2**: Use volume mounting (recommended for large datasets)
     * Don't include data in the container
     * Mount the data directory when running the container

### 2. Create a Dockerfile for MLflow

Create a file named `Dockerfile.mlflow` in your project root (see the example how it is done in the Flowers Classification App repo):

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /mlflow

# Install MLflow
RUN pip install --no-cache-dir mlflow pymysql

COPY 

# Expose the default MLflow UI port
EXPOSE

# Command to run MLflow server
CMD 
```

### 3. Create a Docker Compose File

Create a `docker-compose.yml` file in your project root. This file will define the services for both your ML application and MLflow (see the example how it is done in the Flowers Classification App repo):

```yaml
version: '3'

services:
  ml-app:
    build:
      context: .
      dockerfile: Dockerfile.mlapp
    ...
  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    ...

networks:
  ml-network:
    driver: bridge
```

**Why Docker Compose?**

Docker Compose is essential for this project because:
1. **Service Orchestration**: Manages startup order (ensuring MLflow is running before your ML app)
2. **Networking**: Creates an internal network allowing containers to communicate
3. **Configuration Management**: Centralizes environment variables and port mappings
4. **Volume Management**: Simplifies mounting directories for data persistence

With Docker Compose, MLflow and your ML application can communicate seamlessly. Your ML application can log metrics, parameters, and models to MLflow running in a separate container. This is achieved through the internal network named `ml-network` with `bridge` driver in the compose file.

### 4. Updating Your Code for MLflow Integration

Ensure your training script is configured to connect to the MLflow tracking server:

```python
# In your train.py
import os
import mlflow

# Get the tracking URI from environment variable or use default
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# The rest of your training code with MLflow tracking
```

### 5. Building and Running Your Containers

Build and start your containers:

```bash
docker-compose up --build
```

This command builds both Docker images and starts the containers. Your services will be available at:
- ML Application API: http://localhost:(your-api-port)
- MLflow UI: http://localhost:(your-mlflow-port)

### 6. Publishing Your Docker Images

Once your containerized application is working correctly, publish your images to Docker Hub:

1. **Log in to Docker Hub**:
   ```bash
   docker login
   ```
   
2. **Tag your images**:
   ```bash
   docker tag <your-local-ml-app-image> <your-dockerhub-username>/ml-application:latest
   docker tag <your-local-mlflow-image> <your-dockerhub-username>/mlflow-tracking:latest
   ```
   
3. **Push your images**:
   ```bash
   docker push <your-dockerhub-username>/ml-application:latest
   docker push <your-dockerhub-username>/mlflow-tracking:latest
   ```

## Deliverables

Submit the following:

1. **GitHub Repository**:
   * URL to your repository containing:
     * All project code following the structure from previous assignments
     * `Dockerfile.mlapp`
     * `Dockerfile.mlflow`
     * `docker-compose.yml`
     * Updated README with instructions for running your containerized application

2. **Docker Hub Links**:
   * URL to your ML application image
   * URL to your MLflow image

3. **Team Contribution**:
   * Commit history should demonstrate contributions from all team members
   * Include a brief description of each member's contribution to the containerization process

## Implementing Logging in Your Containerized Application

Proper logging is essential for monitoring and troubleshooting containerized applications. For this assignment, you will implement the Python `logging` package in your application to create structured logs for your Docker containers.

### Python Logging Basics

The Python `logging` module provides a flexible framework for emitting log messages from applications. Here's how to implement it directly in your application:

```python
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app.log')  # Log to file
    ]
)

# Create a logger for a specific module
logger = logging.getLogger(__name__)

# Example usage
def predict(data):
    try:
        logger.info(f"Received prediction request with shape: {data.shape}")
        result = model.predict(data)
        logger.info(f"Prediction successful")
        return result
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        raise
```

### Logging from Different Modules

Structure your logging configuration to separate logs by module:

```python
# In src/logging_config.py <<<If you want to perform this approach, you will need to create this file in src folder>>>
import logging
import os

def configure_logging(log_directory='logs'):
    # Create logs directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create module-specific loggers
    modules = ['train', 'predict', 'api']
    loggers = {}
    
    for module in modules:
        logger = logging.getLogger(f'ml_app.{module}')
        file_handler = logging.FileHandler(f'{log_directory}/{module}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        loggers[module] = logger
    
    return loggers
```

### Using the Loggers in Your Application

In your application modules you can directly do (this assumes you have created logging_config.py in src folder):

```python
# In src/train.py
from logging_config import configure_logging

loggers = configure_logging()
logger = loggers['train']

def train_model():
    logger.info("Starting model training")
    # Training code here
    logger.info(f"Model trained successfully with accuracy: {accuracy}")
```

### Docker Logging Integration

When running applications in Docker containers, it's best practice to:  

1. **Log to stdout/stderr**: Docker captures these streams

   ```python
   # Configure logging to write to stdout
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[logging.StreamHandler()]  # Log to stdout
   )
   ```

2. **Modify your Dockerfile to handle logs**:

   ```dockerfile
   # Create a logs directory in the container
   RUN mkdir -p /app/logs
   
   # Set an environment variable for log location
   ENV LOG_DIR=/app/logs
   
   # Volume mount for persistent logs (in docker-compose.yml)
   volumes:
     - ./logs:/app/logs
   ```

3. **Implement log rotation** (this helps to keep log at a controllable size):

   ```python
   from logging.handlers import RotatingFileHandler
   
   handler = RotatingFileHandler(
       'app.log',
       maxBytes=10485760,  # 10MB
       backupCount=5
   )
   logger.addHandler(handler)
   ```

### Example: Adding Logging to predict_api.py

Update your `predict_api.py` file to include proper logging:

```python
import logging
import os
from flask import Flask, request, jsonify

# Configure logging
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{log_dir}/api.log")
    ]
)

logger = logging.getLogger("ml_app.api")

app = Flask(__name__)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request (v1) with data: {data}")
        
        # Model prediction code
        result = model_v1.predict([data])
        
        logger.info(f"Prediction (v1) successful: {result}")
        return jsonify({"prediction": result.tolist()})
    except Exception as e:
        logger.error(f"Prediction (v1) failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting prediction API service")
    app.run(host="0.0.0.0", port=5000)
```

## Evaluation Criteria

Your assignment will be evaluated based on:

1. **Functionality**: Both containers work correctly and communicate with each other
2. **Code Quality**: Well-organized and well-documented Dockerfiles and docker-compose.yml
3. **Data Management**: Appropriate strategy for handling data (volume mounting or copying)
4. **MLflow Integration**: Proper configuration allowing your ML application to log to MLflow
5. **Logging Implementation**: Proper implementation of Python logging for monitoring and troubleshooting
6. **Documentation**: Clear instructions on how to build and run your containers
7. **Teamwork**: Evidence of contributions from all team members

## Tips for Success

1. **Test Locally**: Before pushing to Docker Hub, ensure everything works on your local machine
2. **Layer Optimization**: Organize your Dockerfiles to take advantage of layer caching
3. **Version Pinning**: Specify exact versions of base images and dependencies
4. **Security**: Avoid including sensitive information in your Docker images
5. **Documentation**: Include comments in your Dockerfiles explaining key decisions

## Submission Guidelines

Submit your deliverables through the course submission system by the deadline. Ensure all team members have contributed to the GitHub repository.

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)