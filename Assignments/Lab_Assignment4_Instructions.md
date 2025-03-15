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
- ML Application API: http://localhost:5000
- MLflow UI: http://localhost:5001

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

## Evaluation Criteria

Your assignment will be evaluated based on:

1. **Functionality**: Both containers work correctly and communicate with each other
2. **Code Quality**: Well-organized and well-documented Dockerfiles and docker-compose.yml
3. **Data Management**: Appropriate strategy for handling data (volume mounting or copying)
4. **MLflow Integration**: Proper configuration allowing your ML application to log to MLflow
5. **Documentation**: Clear instructions on how to build and run your containers
6. **Teamwork**: Evidence of contributions from all team members

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